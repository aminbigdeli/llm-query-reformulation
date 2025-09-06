import argparse
import math
import csv

def dcg(relevances):
    return sum((rel / math.log2(idx + 2) for idx, rel in enumerate(relevances)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-qrels', type=str, default='')
    parser.add_argument('-run', type=str, default='')
    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-output', type=str, default='')
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
            qid, did, _ = line.strip().split("\t")
            if qid not in run:
                run[qid] = []
            run[qid].append(did)

    ndcg_total = 0.0
    query_scores = []
    
    for qid in run:
        if qid in qrel:
            rels = [qrel[qid].get(did, 0) for did in run[qid][:k]]
            dcg_val = dcg(rels)
            ideal_rels = sorted(qrel[qid].values(), reverse=True)[:k]
            idcg_val = dcg(ideal_rels)
            ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0
        else:
            ndcg = 0.0
        ndcg_total += ndcg
        query_scores.append((qid, ndcg))
    
    ndcg_total /= len(run)
    print(f"NDCG@{k}: {ndcg_total}")
    
    # Save individual query scores to CSV if output path is provided
    if args.output:
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['qid', f'ndcg@{k}'])
            for qid, score in query_scores:
                writer.writerow([qid, score])
        print(f"Individual query scores saved to: {args.output}")

if __name__ == "__main__":
    main() 