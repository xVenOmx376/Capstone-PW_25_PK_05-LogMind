"""DRY-RUN ONLY: generates a small HDFS-style log + labels so you can test the pipeline
before downloading the real dataset. Results on this data are NOT results for your project.
For the real review, put HDFS.log and anomaly_label.csv from LogHub in data/ instead."""
import random, datetime
from config import *

r = random.Random(7)
N_NORMAL, N_ANOM = 5200, 450
BASE = datetime.datetime(2008, 11, 9, 20, 35, 18)


def ip(): return f"10.250.{r.randint(10, 19)}.{r.randint(10, 250)}"
def bid(): return f"blk_{'-' if r.random() < 0.5 else ''}{r.randrange(10**18, 9 * 10**18)}"


def normal(b):
    ips = [ip() for _ in range(3)]
    path = f"/user/root/rand/_temporary/_task_2008110910{r.randint(10,99)}_{r.randint(1,9):04d}_m_{r.randint(1,999):06d}_0/part-{r.randint(1,999):05d}"
    L = [(0, "INFO", "dfs.FSNamesystem", f"BLOCK* NameSystem.allocateBlock: {path}. {b}")]
    for i, x in enumerate(ips):
        L.append((1 + i, "INFO", "dfs.DataNode$DataXceiver", f"Receiving block {b} src: /{x}:{r.randint(30000,60000)} dest: /{x}:50010"))
    for i, x in enumerate(ips):
        L.append((4 + i, "INFO", "dfs.DataNode$PacketResponder", f"PacketResponder {2 - i} for block {b} terminating"))
        L.append((4 + i, "INFO", "dfs.DataNode$DataXceiver", f"Received block {b} of size 67108864 from /{x}"))
        L.append((5 + i, "INFO", "dfs.FSNamesystem", f"BLOCK* NameSystem.addStoredBlock: blockMap updated: {x}:50010 is added to {b} size 67108864"))
    if r.random() < 0.06:  # routine background scan: rare template, but NORMAL
        L.append((r.randint(8, 40), "INFO", "dfs.DataBlockScanner", f"Verification succeeded for {b}"))
    return L


def anomaly(b):
    L = normal(b)
    kind = r.choice("abcde")
    if kind == "a":
        L = [l for l in L if "PacketResponder 0" not in l[3] and "Received block" not in l[3]][:-2]
        L.append((6, "WARN", "dfs.DataNode$DataXceiver", f"Exception in receiveBlock for block {b} java.io.IOException: Connection reset by peer"))
        L.append((7, "INFO", "dfs.DataNode$DataXceiver", f"writeBlock {b} received exception java.io.IOException: Connection reset by peer"))
    elif kind == "b":
        L.append((7, "WARN", "dfs.DataNode$PacketResponder", f"PacketResponder {b} 1 Exception java.io.IOException: Broken pipe"))
        L.append((9, "WARN", "dfs.FSDataset", f"Unexpected error trying to delete block {b}. BlockInfo not found in volumeMap."))
    elif kind == "c":
        L.append((8, "WARN", "dfs.FSNamesystem", f"Redundant addStoredBlock request received for {b} on {ip()}:50010 size 67108864"))
        L.append((9, "INFO", "dfs.FSNamesystem", f"BLOCK* ask {ip()}:50010 to replicate {b} to datanode(s) {ip()}:50010"))
    elif kind == "d":
        L.append((10, "WARN", "dfs.DataNode$DataXceiver", f"{ip()}:50010:Got exception while serving {b} to /{ip()}: java.net.SocketTimeoutException"))
        L.append((12, "INFO", "dfs.FSNamesystem", f"BLOCK* NameSystem.delete: {b} is added to invalidSet of {ip()}:50010"))
    else:  # subtle: one replica silently missing its final steps
        L = L[:-2]
    return L


events, labels = [], []
for i in range(N_NORMAL + N_ANOM):
    b, is_a = bid(), i >= N_NORMAL
    labels.append((b, "Anomaly" if is_a else "Normal"))
    start = r.randint(0, 3 * 3600)
    for off, lvl, comp, msg in (anomaly(b) if is_a else normal(b)):
        events.append((start + off + r.random(), lvl, comp, msg))
events.sort(key=lambda e: e[0])
with open(RAW_LOG, "w") as f:
    for t, lvl, comp, msg in events:
        ts = BASE + datetime.timedelta(seconds=int(t))
        f.write(f"{ts:%y%m%d} {ts:%H%M%S} {r.randint(10, 200)} {lvl} {comp}: {msg}\n")
with open(RAW_LABELS, "w") as f:
    f.write("BlockId,Label\n")
    f.writelines(f"{b},{l}\n" for b, l in labels)
SYNTHETIC_FLAG.write_text("synthetic")
print(f"SYNTHETIC data written: {len(events)} lines, {N_NORMAL} normal + {N_ANOM} anomalous blocks")
