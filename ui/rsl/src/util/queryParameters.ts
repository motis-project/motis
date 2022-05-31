export default function getQueryParameters(): URLSearchParams {
  if (typeof window !== "undefined") {
    return new URL(window.location.href).searchParams;
  } else {
    return new URLSearchParams();
  }
}
