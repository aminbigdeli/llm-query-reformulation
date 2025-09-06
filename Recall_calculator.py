import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-qrels', type=str, default='')
    parser.add_argument('-run', type=str, default='')
    parser.add_argument('-metric', type=str, default='recall_cut_1000')
    parser.add_argument('-result', type=str, default='')
    args = parser.parse_args()

    metric = args.metric
    k = int(metric.split('_')[-1])
    
    qrel = {}
    with open(args.qrels, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)

    run = {}
    with open(args.run, 'r') as f_run:
        for line in f_run:
            # qid, _, did, _, _, _ = line.strip().split()
            qid, did, _ = line.strip().split("\t")
            if qid not in run: 
                run[qid] = []
            run[qid].append(did)

    recall = 0.0
    for qid in run:
        # Count relevant documents in top-k
        relevant_in_topk = 0
        total_relevant = 0
        
        # Count total relevant documents for this query
        if qid in qrel:
            total_relevant = sum(1 for label in qrel[qid].values() if label > 0)
        
        # Count relevant documents in top-k
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                relevant_in_topk += 1
        
        # Calculate recall for this query
        if total_relevant > 0:
            query_recall = relevant_in_topk / total_relevant
        else:
            query_recall = 0.0
            
        recall += query_recall
    
    recall /= len(run)
    print(f"Recall@{k}: {recall}")


if __name__ == "__main__":
    main()