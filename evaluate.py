import json, time, csv
from pathlib import Path
from v2_faiss_rag import ask as ask_faiss
from v3_chroma_rag import ask as ask_chroma
from v4_reranking import ask as ask_rerank

def run_one(version, fn, question):
    start=time.perf_counter()
    try:
        result=fn(question); elapsed=round(time.perf_counter()-start,3)
        if version=='V4_RERANKING': answer,sources,candidates,selected=result; docs=selected
        else: answer,sources,docs=result
        return {'version':version,'question':question,'answer':answer,'response_time_seconds':elapsed,
                'source_pages':'; '.join(f"{s['source']} p.{s['page']}" for s in sources),
                'retrieved_chunk_count':len(docs),'error':''}
    except Exception as e:
        return {'version':version,'question':question,'answer':'','response_time_seconds':round(time.perf_counter()-start,3),
                'source_pages':'','retrieved_chunk_count':0,'error':f'{type(e).__name__}: {e}'}

def main():
    questions=json.loads(Path('evaluation_questions.json').read_text(encoding='utf-8')); rows=[]
    for i,item in enumerate(questions,1):
        print(f"[{i}/{len(questions)}] {item['question']}")
        for name,fn in [('V2_FAISS',ask_faiss),('V3_CHROMA',ask_chroma),('V4_RERANKING',ask_rerank)]:
            rows.append(run_one(name,fn,item['question']))
    with open('evaluation_results.csv','w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print('Wrote evaluation_results.csv. Manually score accuracy and groundedness against the expected answers.')
if __name__=='__main__': main()
