"""
MVP Baseline Benchmark Script

Tests the current implemented stack:
- Chunking: paragraph-level (from benchmark_input.csv)
- Embedding: NVIDIA nv-embedqa-e5-v5
- Vector DB: Milvus
- Index: IVF_FLAT
- Retrieval: COSINE
- Reranking: none

Usage:
    python scripts/run_baseline_benchmark.py --limit 10
    python scripts/run_baseline_benchmark.py --limit 0  # all rows
"""

import os
import sys
import csv
import time
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing services
from source.services.embedding_service import embedding_service
from source.services.vector_service import get_vector_service

# Configuration
INPUT_CSV = Path("data/benchmark_input.csv")
OUTPUT_CSV = Path("data/baseline_benchmark_results.csv")
COLLECTION_NAME = "baseline_benchmark"

# Benchmark configuration
BENCHMARK_CONFIG = {
    "chunking_method": "paragraph_from_csv",
    "embedding_model": "nvidia/nv-embedqa-e5-v5",
    "vector_database": "Milvus",
    "index_type": "IVF_FLAT",
    "retrieval_method": "COSINE",
    "reranking_model": "none",
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_benchmark_data(limit: int = 0) -> List[Dict[str, Any]]:
    """Load benchmark input CSV."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    
    rows = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'id': int(row['id']),
                'pdf_name': row['pdf_name'],
                'paragraph': row['paragraph']
            })
            if limit > 0 and len(rows) >= limit:
                break
    
    logger.info(f"Loaded {len(rows)} rows from {INPUT_CSV}")
    return rows


async def check_milvus_connection(vector_service) -> bool:
    """Check if Milvus is accessible."""
    try:
        vector_service.ensure_connected()
        if not vector_service.connected:
            logger.error("=" * 60)
            logger.error("MILVUS CONNECTION FAILED")
            logger.error("=" * 60)
            logger.error("Milvus is not running or not accessible.")
            logger.error("")
            logger.error("Next steps:")
            logger.error("1. Check if Milvus is running:")
            logger.error("   docker ps | grep milvus")
            logger.error("")
            logger.error("2. Start Milvus if not running:")
            logger.error("   docker run -d --name milvus-standalone \\")
            logger.error("     -p 19530:19530 -p 9091:9091 \\")
            logger.error("     milvusdb/milvus:latest")
            logger.error("")
            logger.error("3. Check config/QA_Text/config.yaml:")
            logger.error("   milvus_host: localhost")
            logger.error("   milvus_port: 19530")
            logger.error("")
            logger.error("4. Or use milvus-lite (embedded):")
            logger.error("   pip install milvus-lite")
            logger.error("=" * 60)
            return False
        
        logger.info("✓ Milvus connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Milvus connection error: {e}")
        return False


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using existing embedding service."""
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    start = time.time()
    
    try:
        responses = await embedding_service.create_embeddings_batch(
            texts=texts,
            batch_size=10
        )
        
        embeddings = [r.embedding for r in responses]
        duration = time.time() - start
        
        logger.info(f"✓ Generated {len(embeddings)} embeddings in {duration:.2f}s")
        return embeddings
    
    except Exception as e:
        logger.error(f"✗ Embedding generation failed: {e}")
        raise


async def store_embeddings_in_milvus(
    vector_service,
    rows: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> bool:
    """Store embeddings in Milvus."""
    logger.info(f"Storing {len(embeddings)} embeddings in Milvus...")
    
    try:
        # Prepare data for Milvus
        texts = [row['paragraph'] for row in rows]
        metadata = [
            {
                'benchmark_id': row['id'],
                'pdf_name': row['pdf_name'],
                'text_length': len(row['paragraph'])
            }
            for row in rows
        ]
        
        # Store in Milvus
        success = await vector_service.store_document_embeddings(
            document_id="baseline_benchmark",
            embeddings=embeddings,
            texts=texts,
            metadata=metadata,
            collection_name=COLLECTION_NAME
        )
        
        if success:
            logger.info(f"✓ Stored embeddings in collection '{COLLECTION_NAME}'")
        else:
            logger.error(f"✗ Failed to store embeddings")
        
        return success
    
    except Exception as e:
        logger.error(f"✗ Storage failed: {e}")
        raise


async def run_retrieval_queries(
    vector_service,
    rows: List[Dict[str, Any]],
    embeddings: List[List[float]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Run retrieval queries and collect results."""
    logger.info(f"Running {len(rows)} retrieval queries (top_k={top_k})...")
    
    results = []
    
    for i, (row, query_embedding) in enumerate(zip(rows, embeddings)):
        query_id = row['id']
        
        try:
            # Measure retrieval latency
            start = time.time()
            
            search_results = await vector_service.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                collection_name=COLLECTION_NAME
            )
            
            retrieval_latency_ms = (time.time() - start) * 1000
            
            # Extract retrieved IDs and scores
            retrieved_ids = []
            retrieved_scores = []
            
            for result in search_results:
                # Get benchmark_id from metadata
                meta = result.metadata or {}
                benchmark_id = meta.get('benchmark_id', '')
                retrieved_ids.append(str(benchmark_id))
                retrieved_scores.append(f"{result.score:.4f}")
            
            # Create result record
            result_record = {
                'query_id': query_id,
                'pdf_name': row['pdf_name'],
                'chunking_method': BENCHMARK_CONFIG['chunking_method'],
                'embedding_model': BENCHMARK_CONFIG['embedding_model'],
                'vector_database': BENCHMARK_CONFIG['vector_database'],
                'index_type': BENCHMARK_CONFIG['index_type'],
                'retrieval_method': BENCHMARK_CONFIG['retrieval_method'],
                'reranking_model': BENCHMARK_CONFIG['reranking_model'],
                'top_k': top_k,
                'retrieved_ids': '|'.join(retrieved_ids),
                'retrieved_scores': '|'.join(retrieved_scores),
                'retrieval_latency_ms': f"{retrieval_latency_ms:.2f}",
                'error': '',
                'notes': ''
            }
            
            results.append(result_record)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(rows)} queries")
        
        except Exception as e:
            logger.error(f"✗ Query {query_id} failed: {e}")
            
            # Record error
            result_record = {
                'query_id': query_id,
                'pdf_name': row['pdf_name'],
                'chunking_method': BENCHMARK_CONFIG['chunking_method'],
                'embedding_model': BENCHMARK_CONFIG['embedding_model'],
                'vector_database': BENCHMARK_CONFIG['vector_database'],
                'index_type': BENCHMARK_CONFIG['index_type'],
                'retrieval_method': BENCHMARK_CONFIG['retrieval_method'],
                'reranking_model': BENCHMARK_CONFIG['reranking_model'],
                'top_k': top_k,
                'retrieved_ids': '',
                'retrieved_scores': '',
                'retrieval_latency_ms': '',
                'error': str(e),
                'notes': 'Query failed'
            }
            
            results.append(result_record)
    
    logger.info(f"✓ Completed {len(results)} queries")
    return results


def write_results(results: List[Dict[str, Any]]) -> None:
    """Write results to CSV."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'query_id',
        'pdf_name',
        'chunking_method',
        'embedding_model',
        'vector_database',
        'index_type',
        'retrieval_method',
        'reranking_model',
        'top_k',
        'retrieved_ids',
        'retrieved_scores',
        'retrieval_latency_ms',
        'error',
        'notes'
    ]
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"✓ Results written to {OUTPUT_CSV}")


async def cleanup_collection(vector_service) -> None:
    """Clean up benchmark collection."""
    try:
        logger.info(f"Cleaning up collection '{COLLECTION_NAME}'...")
        await vector_service.drop_collection(COLLECTION_NAME)
        logger.info("✓ Collection cleaned up")
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")


async def run_benchmark(limit: int = 10, top_k: int = 5, cleanup: bool = True) -> None:
    """Run the baseline benchmark."""
    logger.info("=" * 60)
    logger.info("BASELINE BENCHMARK - MVP")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    for key, value in BENCHMARK_CONFIG.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  top_k: {top_k}")
    logger.info(f"  limit: {limit if limit > 0 else 'all'}")
    logger.info("=" * 60)
    
    # Get vector service
    vector_service = get_vector_service()
    if not vector_service:
        logger.error("✗ Vector service not available")
        return
    
    # Check Milvus connection
    if not await check_milvus_connection(vector_service):
        return
    
    try:
        # Load data
        rows = load_benchmark_data(limit=limit)
        
        if not rows:
            logger.error("✗ No data loaded")
            return
        
        # Generate embeddings
        texts = [row['paragraph'] for row in rows]
        embeddings = await generate_embeddings(texts)
        
        if len(embeddings) != len(rows):
            logger.error(f"✗ Embedding count mismatch: {len(embeddings)} != {len(rows)}")
            return
        
        # Store embeddings
        success = await store_embeddings_in_milvus(vector_service, rows, embeddings)
        if not success:
            logger.error("✗ Failed to store embeddings")
            return
        
        # Wait a moment for indexing (milvus-lite needs time)
        logger.info("Waiting for indexing...")
        await asyncio.sleep(2)
        
        # Verify data was stored
        try:
            count = await vector_service.count_documents(COLLECTION_NAME)
            logger.info(f"Collection has {count} documents")
        except Exception as e:
            logger.warning(f"Could not count documents: {e}")
        
        # Run retrieval queries
        results = await run_retrieval_queries(vector_service, rows, embeddings, top_k=top_k)
        
        # Write results
        write_results(results)
        
        # Summary
        logger.info("=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total queries: {len(results)}")
        
        successful = sum(1 for r in results if not r['error'])
        failed = len(results) - successful
        
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        
        if successful > 0:
            latencies = [float(r['retrieval_latency_ms']) for r in results if r['retrieval_latency_ms']]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                logger.info(f"Avg retrieval latency: {avg_latency:.2f}ms")
                logger.info(f"Min retrieval latency: {min_latency:.2f}ms")
                logger.info(f"Max retrieval latency: {max_latency:.2f}ms")
        
        logger.info(f"Output: {OUTPUT_CSV}")
        logger.info("=" * 60)
        logger.info("✓ Benchmark complete!")
        
    except Exception as e:
        logger.error(f"✗ Benchmark failed: {e}", exc_info=True)
    
    finally:
        # Cleanup
        if cleanup:
            await cleanup_collection(vector_service)


def main():
    parser = argparse.ArgumentParser(description="Run baseline benchmark")
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of rows to process (0 for all, default: 10)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to retrieve per query (default: 5)'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Do not clean up the benchmark collection after completion'
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(
        limit=args.limit,
        top_k=args.top_k,
        cleanup=not args.no_cleanup
    ))


if __name__ == "__main__":
    main()
