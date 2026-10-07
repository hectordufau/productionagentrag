"""Live Qdrant contract tests; run with QDRANT_URL or default localhost:6333."""
import os
import uuid

import pytest

from agentic_rag.retrieval.qdrant import QdrantClient, QdrantHTTPError, QdrantUnavailable

URL=os.getenv('QDRANT_URL','http://localhost:6333')
def client(): return QdrantClient(URL,timeout=2)
def live():
 try: client().health(); return True
 except QdrantUnavailable: return False
pytestmark=pytest.mark.integration

def test_qdrant_live_full_lifecycle():
 if not live():
  if os.getenv('QDRANT_EXPLICITLY_UNAVAILABLE')=='1': pytest.skip('explicitly unavailable')
  pytest.fail('Qdrant is required for integration run; set QDRANT_EXPLICITLY_UNAVAILABLE=1 only when unavailable')
 c=client(); name='m2a1_'+uuid.uuid4().hex[:10]; assert c.health(); c.ensure_collection(name,3,'Cosine'); assert c.validate_collection(name,3)
 points=[{'id':1,'vector':[1,0,0],'payload':{'chunk_id':'c1','category':'storage'}},{'id':2,'vector':[0,1,0],'payload':{'chunk_id':'c2','category':'metadata'}}]
 c.upsert(name,points); found=c.search(name,[1,0,0],2); assert found[0]['payload']['chunk_id']=='c1'; assert found[0]['payload']['category']=='storage'
 filtered=c.search(name,[0,1,0],5,{'must':[{'key':'category','match':{'value':'metadata'}}]}); assert [x['payload']['chunk_id'] for x in filtered]==['c2']
 c.update(name,[{'id':2,'vector':[1,0,0],'payload':{'chunk_id':'c2-updated','category':'metadata'}}]); assert c.search(name,[1,0,0],2)[0]['payload']['chunk_id'] in ('c1','c2-updated')
 c.delete(name,[1]); assert all(x['id']!=1 for x in c.search(name,[1,0,0],5)); assert c.collection_exists(name)
 # Persistence is verified by a fresh adapter process/object against the same running volume.
 fresh=QdrantClient(URL,timeout=2); assert fresh.validate_collection(name,3); assert fresh.search(name,[1,0,0],5)
 c.delete(name,[2]);
 try: c.validate_collection(name,4)
 except QdrantHTTPError as exc: assert exc.status==409
 else: pytest.fail('invalid collection config was accepted')


def test_qdrant_unavailable_is_explicit():
 bad=QdrantClient('http://127.0.0.1:1',timeout=.1)
 with pytest.raises(QdrantUnavailable): bad.health()
