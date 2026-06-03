# WNS Reranker Smoke Results — 2026-05-28

## BGE reranker smoke

Purpose: prove retrieved vector hits can be passed to a real reranker endpoint and reordered. This is not Recall/MRR/nDCG evaluation.

Input:

```text
query: refund old ticket and issue new ticket
sheet/chunker: fixed_tok1200_ov150
embedding: gte_multilingual_base
vector_db: Qdrant
retrieval_top_k: 20
reranker: bge_reranker_base
```

Timing:

```text
rerank_seconds: 0.103
```

### Top 5 before rerank

```text
1. score=0.7793 | Content - Ongoing Issues & Errors 20250310-073922.pdf
2. score=0.7761 | Content - TUI Flybe (TB) 20250310-074844.pdf
3. score=0.7655 | Naco_ Scenario of refund of old ticket and then issuing a new ticket.pdf
4. score=0.7650 | Content - Farelogix LH Group - step by step for FL tasks 20250310-073022.pdf
5. score=0.7590 | 07. New - Handling NACO request for CC2C orders.pdf
```

### Top 5 after BGE rerank

```text
1. score=0.9965 | Content - Farelogix LH Group - step by step for FL tasks 20250310-073022.pdf
   Preview: Refund only the EMD: Baggage and seats have to be refunded via the BSP link. If you have proceeded with a reissue, but the airline is unable to associate the EMD to the new reissued ticket, then we need to refund the old

2. score=0.9944 | 07. New - Handling NACO request for CC2C orders.pdf
   Preview: The information in this blog may change at any time post blog creation date, always refer to the actual process/routine link provided for the most up-to-date information. We have established a new process for NACO reques

3. score=0.9907 | Naco_ Scenario of refund of old ticket and then issuing a new ticket.pdf
   Preview: After checking the Shelf page: How to handle naco/nach - General Rules, the FL agent should continue with the below steps for the scenario of refunding the old ticket and then issuing a new ticket. 1. FL agents should ch

4. score=0.9826 | Content - Ongoing Issues & Errors 20250310-073922.pdf
   Preview: AI Refund against fare rule. Issue new ticket in same PNR. The name ﬁeld does not need to be changed, they don't see CHD. BT No need to contact BT. Make new PNR and issue new adt tkt. Refund on old tkt according to rule.

5. score=0.9752 | Content - TUI Flybe (TB) 20250310-074844.pdf
   Preview: Change the status of the order as per the normal process and link the new PNR in Edvin. Refunding the original ticket value after the issuance of the new due to SC- second alternative provided. After the new ticket is is
```

Result:

```text
BGE reranker endpoint works on real Qdrant retrieved KnowRA chunks.
Reranker reordered initial vector hits into a more directly refund/reissue-focused order.
```

## Qwen reranker smoke

Purpose: prove the Qwen reranker endpoint can rerank real Qdrant retrieval results. This is not Recall/MRR/nDCG evaluation.

Input:

```text
query: refund old ticket and issue new ticket
sheet/chunker: fixed_tok1200_ov150
embedding: gte_multilingual_base
vector_db: Qdrant
retrieval_top_k: 20
reranker: qwen3_4b_rerank
```

Timing:

```text
retrieval_seconds: 0.046
rerank_seconds: 1.592
```

### Top 5 before rerank

```text
1. score=0.7793 | Content - Ongoing Issues & Errors 20250310-073922.pdf
2. score=0.7761 | Content - TUI Flybe (TB) 20250310-074844.pdf
3. score=0.7655 | Naco_ Scenario of refund of old ticket and then issuing a new ticket.pdf
4. score=0.7650 | Content - Farelogix LH Group - step by step for FL tasks 20250310-073022.pdf
5. score=0.7590 | 07. New - Handling NACO request for CC2C orders.pdf
```

### Top 5 after Qwen rerank

```text
1. score=0.9977 | Content - Farelogix LH Group - step by step for FL tasks 20250310-073022.pdf
   Preview: EMD/TKT - Refund to us: a new EMD is issued by SC (after a ticket was reissued and it was not possible to associate the EMD), thus the old EMD should be refunded to us. No EMD refund to pax. EMD - Refund to customer: An

2. score=0.9972 | Content - Refunds 20250310-074217.pdf
   Preview: In this case, we do not use the Modify order button as we do not want to create a refund case. 2. Communication over email – ticket expired at/before the time of contact. In case 1, we should explain to the customer that

3. score=0.9946 | Content - Farelogix Emirates (EK) 20250310-073143.pdf
   Preview: Step 3 - Click "Refund" and the ticket refund screen will be displayed: The refund amount needs to be calculated manually, respecting unused

4. score=0.9919 | Content - ETG and Partner Products 20250310-072912.pdf
   Preview: Cancellation A ticket that has been rebooked using the Flexible Ticket will be refunded according to the standard refund process of voluntary reissued tickets. This means that we need to check the rules of both the origi

5. score=0.9901 | 07. New - Handling NACO request for CC2C orders.pdf
   Preview: The information in this blog may change at any time post blog creation date, always refer to the actual process/routine link provided for the most up-to-date information. We have established a new process for NACO reques
```

Result:

```text
Qwen reranker endpoint works on real Qdrant retrieved KnowRA chunks.
Qwen rerank was slower than BGE on this smoke query, 1.592s vs 0.103s, and produced a refund-focused but different top-5 order.
```

Expected artifact on VM from smoke script:

```text
data/reranker_smoke/qwen_fixed_gte_qdrant_refund.json
```
