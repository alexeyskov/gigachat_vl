"""Real flock/mmap regression for audit versus inactive-file guard."""
import mmap,os,subprocess,sys,time
from pathlib import Path
import pytest
from scripts.audit_research_budget18_updates_v1 import audit_locked


@pytest.mark.parametrize('fail',[False,True])
def test_shared_guard_cannot_observe_auditor_mappings_after_exclusive_release(tmp_path,fail):
    (tmp_path/'.checkpoint.lock').write_bytes(b'');data=tmp_path/'owned-state.bin';data.write_bytes(b'tiny trusted checkpoint fixture')
    attempted=tmp_path/'shared_attempt';children=[]
    code='''import fcntl,sys
from pathlib import Path
lock,data,parent,attempt=sys.argv[1:]
with open(lock,'rb') as f:
 Path(attempt).write_text('attempt')
 fcntl.flock(f,fcntl.LOCK_SH)
 assert data not in Path('/proc',parent,'maps').read_text(), 'Mapped audit state survived lock release'
 print('guard_observed_no_mapping')
'''
    def inspect():
        with data.open('rb') as f:
            mapping=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        cycle=[];cycle.append((cycle,mapping))
        child=subprocess.Popen([sys.executable,'-c',code,str(tmp_path/'.checkpoint.lock'),str(data),str(os.getpid()),str(attempted)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        children.append(child);deadline=time.monotonic()+3
        while not attempted.exists():
            assert time.monotonic()<deadline;time.sleep(.01)
        assert str(data) in Path('/proc',str(os.getpid()),'maps').read_text()
        assert child.poll() is None
        if fail:raise ValueError('controlled failure with mapped tensor cycle')
    try:
        if fail:
            with pytest.raises(ValueError,match='controlled failure'):audit_locked(tmp_path,inspect)
        else:audit_locked(tmp_path,inspect)
        stdout,stderr=children[0].communicate(timeout=5)
        assert children[0].returncode==0,stderr
        assert 'guard_observed_no_mapping' in stdout
        assert data.read_bytes()==b'tiny trusted checkpoint fixture'
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait()
