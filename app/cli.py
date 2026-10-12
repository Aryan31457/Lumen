import sys
from .qdrant_store import ensure_collections
from .graphs.ingest_graph import ingest_folder
from .clustering import run_cluster_job
from .reindex import reindex_captions


def main():
    if len(sys.argv) < 2: raise SystemExit('usage: python -m app.cli init|ingest FOLDER|cluster|reindex-captions')
    command=sys.argv[1]
    if command=='init': ensure_collections(); print('collections ready')
    elif command=='ingest' and len(sys.argv)==3: print(ingest_folder(sys.argv[2]))
    elif command=='cluster': print(run_cluster_job())
    elif command=='reindex-captions': print(reindex_captions())
    else: raise SystemExit('usage: python -m app.cli init|ingest FOLDER|cluster|reindex-captions')
if __name__=='__main__': main()
