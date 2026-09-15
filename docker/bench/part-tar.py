#!/usr/bin/env python3
"""Write a tar holding one slice of a file to stdout, without materialising
the slice: part-tar.py <file> <index> <n-parts> <member-name>.

bake-data.sh feeds these to `crane append`, one image layer per slice, so
a 20 GB timetable never needs a second copy on disk.
"""
import io
import os
import sys
import tarfile


class Slice(io.RawIOBase):
    def __init__(self, f, offset, length):
        f.seek(offset)
        self.f = f
        self.left = length

    def readable(self):
        return True

    def readinto(self, b):
        if self.left == 0:
            return 0
        n = self.f.readinto(memoryview(b)[: min(len(b), self.left)])
        self.left -= n
        return n


def main() -> None:
    src, idx, n, member = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    size = os.path.getsize(src)
    part = (size + n - 1) // n
    offset = idx * part
    length = max(0, min(part, size - offset))
    with open(src, "rb") as f, tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
        ti = tarfile.TarInfo(member)
        ti.size = length
        ti.mode = 0o644
        ti.mtime = int(os.path.getmtime(src))
        tar.addfile(ti, io.BufferedReader(Slice(f, offset, length)))


if __name__ == "__main__":
    main()
