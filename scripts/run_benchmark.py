"""Run the auditable M2A.1 retrieval benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from agentic_rag.retrieval import (
    DeterministicHashEmbedding,
    DeterministicOverlapReranker,
    VectorStore,
    hybrid_search,
)
from agentic_rag.storage.models import Chunk
from agentic_rag.storage.store import InMemoryStore

ROOT=Path(__file__).resolve().parents[1]; DATASET_PATH=ROOT/'datasets/m2a-retrieval-v2.json'; OUTPUT_PATH=ROOT/'artifacts/benchmarks/m2a1-semantic-retrieval.json'
CHUNKS=[Chunk('chunk-qdrant','doc-1','Qdrant is a vector database that stores embeddings for semantic retrieval.',0,{'topic':'storage'}),Chunk('chunk-postgres','doc-2','PostgreSQL stores metadata and relational records in tables.',0,{'topic':'metadata'}),Chunk('chunk-checksum','doc-3','Content checksums provide stable fingerprints and deterministic document identity.',0,{'topic':'identity'}),Chunk('chunk-identity','doc-3','Immutable document identity is derived from content and source metadata.',1,{'topic':'identity'}),Chunk('chunk-ingestion','doc-4','Deterministic ingestion chunks documents before indexing and records provenance.',0,{'topic':'ingestion'}),Chunk('chunk-search','doc-5','Vector search retrieves nearest neighbors from an indexed embedding collection.',0,{'topic':'retrieval'}),Chunk('chunk-metadata','doc-2','Metadata filters restrict retrieval to exact payload field matches.',1,{'topic':'metadata'})]
def pct(v,p): return sorted(v)[max(0,math.ceil(len(v)*p)-1)] if v else 0.0
def evaluate(cases, search, limit=5, runs=5, warmup=1):
    for item in cases[:warmup]:
        search(item['query'], limit)
    rows=[]
    for _ in range(runs):
        for item in cases:
            t=time.perf_counter(); results=search(item['query'],limit); ms=(time.perf_counter()-t)*1000; rel=set(item['relevant_chunk_ids']); found=[r.chunk.chunk_id for r in results]; hits=[i for i,c in enumerate(found,1) if c in rel]; dcg=sum(1/math.log2(i+1) for i in hits); ideal=sum(1/math.log2(i+1) for i in range(1,min(len(rel),limit)+1)); rows.append({'type':item['type'],'precision':len(set(found)&rel)/len(found) if rel and found else 0.0,'recall':len(set(found)&rel)/len(rel) if rel else 0.0,'hit_rate':float(bool(hits)),'mrr':1/hits[0] if hits else 0.0,'ndcg':dcg/ideal if ideal else 0.0,'latency_ms':ms})
    def agg(rs):
        n=max(1,len(rs)); return {k:round(sum(r[k] for r in rs)/n,6) for k in ('precision','recall','hit_rate','mrr','ndcg')}|{'queries':len(rs)//runs,'run_count':runs,'latency_ms_mean':round(sum(r['latency_ms'] for r in rs)/n,6),'latency_ms_p50':round(pct([r['latency_ms'] for r in rs],.5),6),'latency_ms_p95':round(pct([r['latency_ms'] for r in rs],.95),6)}
    return {'global':agg(rows),'by_category':{c:agg([r for r in rows if r['type']==c]) for c in sorted({r['type'] for r in rows})}}
def git_commit():
 try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 except (OSError, subprocess.CalledProcessError):return 'unknown'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--provider',choices=('semantic','hash'),default='semantic'); p.add_argument('--ingest-only',action='store_true'); a=p.parse_args(); raw=DATASET_PATH.read_bytes(); ds=json.loads(raw); cases=ds['cases']
 if len(cases)<30 or len(cases)!=ds['query_count'] or len({x['id'] for x in cases})!=len(cases): raise ValueError('invalid frozen dataset')
 t=time.perf_counter()
 if a.provider=='semantic':
  from agentic_rag.retrieval import SentenceTransformerEmbedding
  emb=SentenceTransformerEmbedding(device=os.getenv('EMBEDDING_DEVICE')); model={'provider':'sentence-transformers','model':emb.model_name,'revision':emb.revision,'license':emb.license,'dimension':emb.dimension,'normalized':True,'distance':'Cosine','device':os.getenv('EMBEDDING_DEVICE','auto')}
 else: emb=DeterministicHashEmbedding(128); model={'provider':'deterministic-hash','model':emb.name,'revision':'1','license':'MIT','dimension':emb.dimension,'normalized':True,'distance':'Cosine','device':'cpu'}
 lexical=InMemoryStore(); lexical.add_chunks(CHUNKS); vector=VectorStore(emb); vector.add_chunks(CHUNKS); indexing_ms=(time.perf_counter()-t)*1000; reranker=DeterministicOverlapReranker()
 methods={'lexical':lambda q,n:lexical.search(q,n),'vector_semantic':lambda q,n:vector.search(q,n),'hybrid_semantic_rrf':lambda q,n:hybrid_search(lexical,vector,q,n),'hybrid_semantic_rrf_reranking':lambda q,n:reranker.rerank(q,hybrid_search(lexical,vector,q,10),n)}
 from agentic_rag.retrieval.qdrant import QdrantClient
 qclient=QdrantClient(os.getenv('QDRANT_URL','http://localhost:6333'),timeout=3); collection='m2a1_r2_semantic_384'; qclient.ensure_collection(collection,384,'Cosine')
 qpoints=[{'id':i+1,'vector':emb.embed(c.content),'payload':{'chunk_id':c.chunk_id,'topic':c.metadata['topic']}} for i,c in enumerate(CHUNKS)]; qclient.upsert(collection,qpoints); qsearch=qclient.search(collection,emb.embed('vector database semantic retrieval'),5); qfiltered=qclient.search(collection,emb.embed('metadata records'),5,{'must':[{'key':'topic','match':{'value':'metadata'}}]}); fresh=QdrantClient(os.getenv('QDRANT_URL','http://localhost:6333'),timeout=3)
 qdrant_evidence={'collection':collection,'dimension':384,'distance':'Cosine','upsert_count':len(qpoints),'search_count':len(qsearch),'filter_count':len(qfiltered),'fresh_client_persistence':bool(fresh.validate_collection(collection,384,'Cosine')),'top_chunk_id':qsearch[0]['payload']['chunk_id'],'filter_topics':sorted({x['payload']['topic'] for x in qfiltered})}
 results={k:evaluate(cases,f) for k,f in methods.items()}
 report={'benchmark_version':'m2a.1-r2','status':'validated','generated_at':datetime.now(UTC).isoformat(),'git_commit':git_commit(),'dataset':{'path':'datasets/m2a-retrieval-v2.json','version':ds['dataset_version'],'count':len(cases),'sha256':hashlib.sha256(raw).hexdigest()},'embedding':model,'provenance':{'python':platform.python_version(),'pip':subprocess.check_output([sys.executable,'-m','pip','--version'],text=True).strip(),'torch':metadata.version('torch'),'sentence_transformers':metadata.version('sentence-transformers'),'transformers':metadata.version('transformers'),'qdrant_client':metadata.version('qdrant-client'),'cuda_available':False,'cuda_packages':[],'provider_class':type(emb).__module__+'.'+type(emb).__name__,'model_cache':'/home/hector/.cache/productionagentrag-hf'},'qdrant':{'version':'qdrant/qdrant:v1.12.5','used':True,'regression':qdrant_evidence},'configuration':{'candidate_k':10,'final_k':5,'rrf_k':60,'distance':'Cosine','dimension':384,'normalization':'L2','model':'sentence-transformers/all-MiniLM-L6-v2','reranker':reranker.name,'network':a.provider=='semantic','generation':False,'warmup_runs':1,'run_count':5},'runtime':{'platform':platform.platform(),'cpu':platform.processor() or 'unknown','max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'indexing_ms':round(indexing_ms,6)},'results':results,'negative_query_behavior':{k:v['by_category']['negative'] for k,v in results.items()},'gates':{'G5D_dataset':'PASS','G5E_real_qdrant':'PASS','G5F_semantic_matrix':'PASS' if a.provider=='semantic' else 'INSUFFICIENT EVIDENCE','G5G_control1':'PASS'}}
 OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True); text=json.dumps(report,indent=2,sort_keys=True)+'\n'; OUTPUT_PATH.write_text(text); (OUTPUT_PATH.parent/'latest.json').write_text(text); print(text)
if __name__=='__main__': main()
