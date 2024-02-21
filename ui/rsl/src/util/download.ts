export function getSvgBlob(svgEl: SVGSVGElement) {
  const serializer = new XMLSerializer();
  const source = serializer.serializeToString(svgEl);
  return new Blob([source], { type: "image/svg+xml;charset=utf-8" });
}

export function downloadBlob(url: string, filename: string, release = true) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  if (release) {
    URL.revokeObjectURL(url);
  }
}

export function saveAsSVG(svgEl: SVGSVGElement | null, baseFileName: string) {
  if (!svgEl) {
    return;
  }
  const svgBlob = getSvgBlob(svgEl);
  const url = URL.createObjectURL(svgBlob);
  downloadBlob(url, baseFileName + ".svg");
}

export function saveAsPNG(svgEl: SVGSVGElement | null, baseFileName: string) {
  if (!svgEl) {
    return;
  }
  const svgBlob = getSvgBlob(svgEl);
  const svgUrl = URL.createObjectURL(svgBlob);
  const svgBB = svgEl.getBoundingClientRect();
  const canvas = document.createElement("canvas");
  canvas.width = svgBB.width * 2;
  canvas.height = svgBB.height * 2;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  const img = new Image();
  img.onload = () => {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(svgUrl);
    const pngUrl = canvas.toDataURL("image/png");
    downloadBlob(pngUrl, baseFileName + ".png");
  };
  img.src = svgUrl;
}

export function saveAsCSV(content: string, fileName: string) {
  const blob = new Blob([content], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  downloadBlob(url, fileName);
}
