const BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:8000'

export async function ingestFiles(files: File[] | FileList) {
  const form = new FormData()
  const list: File[] = Array.isArray(files) ? (files as File[]) : Array.from(files as FileList)
  list.forEach((f) => form.append('files', f))
  const res = await fetch(`${BASE}/ingest`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function queryApi(body: any) {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
