export default function Baureihe({ baureihe }: { baureihe: string }) {
  const m = /^[ITR](\d{3})([0-9A-Z])$/.exec(baureihe);
  if (m) {
    return (
      <span title={baureihe}>
        {m[1]}.{m[2]}
      </span>
    );
  } else {
    return <span>{baureihe}</span>;
  }
}
