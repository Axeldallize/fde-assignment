import React, { useState } from 'react'
import { ingestFiles, queryApi } from './api'

export default function App() {
  const [files, setFiles] = useState<FileList | null>(null)
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(12)
  const [useSemantic, setUseSemantic] = useState(true)
  const [useRrf, setUseRrf] = useState(false)
  const [threshold, setThreshold] = useState(0.28)
  const [evidenceTopK, setEvidenceTopK] = useState(4)
  const [temperature, setTemperature] = useState(0.1)
  const [mode, setMode] = useState<'auto' | 'qa' | 'list' | 'table'>('auto')
  const [answer, setAnswer] = useState<string>('')
  const [citations, setCitations] = useState<any[]>([])
  const [meta, setMeta] = useState<any>({})
  const [busy, setBusy] = useState(false)

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setFiles(e.dataTransfer.files)
  }

  const onUpload = async () => {
    if (!files || files.length === 0) return
    setBusy(true)
    try {
      const res = await ingestFiles(files)
      alert(`Ingested: ${res.ingested.join(', ')} (chunks=${res.chunks})`)
    } catch (e: any) {
      alert(`Upload failed: ${e?.message || e}`)
    } finally {
      setBusy(false)
    }
  }

  const onAsk = async () => {
    if (!query.trim()) return
    setBusy(true)
    try {
      const res = await queryApi({
        query,
        mode,
        top_k: topK,
        semantic: useSemantic,
        use_rrf: useRrf,
        evidence_threshold: threshold,
        evidence_topk: evidenceTopK,
        temperature,
      })
      setAnswer(res.answer || res.error || '')
      setCitations(res.citations || [])
      setMeta(res.meta || {})
    } catch (e: any) {
      alert(`Query failed: ${e?.message || e}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ fontFamily: 'Inter, system-ui, Arial', padding: 24, maxWidth: 980, margin: '0 auto' }}>
      <h2>RAG UI</h2>

      <section style={{ marginBottom: 24, padding: 12, border: '1px solid #ddd', borderRadius: 8 }}>
        <h3>Upload PDFs</h3>
        <div
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
          style={{ padding: 24, border: '2px dashed #aaa', borderRadius: 8, marginBottom: 12 }}
        >
          Drag & drop files here
        </div>
        <input type="file" multiple accept="application/pdf" onChange={(e) => setFiles(e.target.files)} />
        <button onClick={onUpload} disabled={busy} style={{ marginLeft: 12 }}>Ingest</button>
      </section>

      <section style={{ marginBottom: 24, padding: 12, border: '1px solid #ddd', borderRadius: 8 }}>
        <h3>Ask</h3>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask a question" style={{ flex: 1 }} />
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="auto">auto</option>
            <option value="qa">qa</option>
            <option value="list">list</option>
            <option value="table">table</option>
          </select>
          <button onClick={onAsk} disabled={busy}>Send</button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 8 }}>
          <label>top_k <input type="number" value={topK} min={1} onChange={(e) => setTopK(parseInt(e.target.value, 10))} /></label>
          <label>semantic <input type="checkbox" checked={useSemantic} onChange={(e) => setUseSemantic(e.target.checked)} /></label>
          <label>use_rrf <input type="checkbox" checked={useRrf} onChange={(e) => setUseRrf(e.target.checked)} /></label>
          <label>threshold <input type="number" step={0.01} value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} /></label>
          <label>evidence_topk <input type="number" min={1} value={evidenceTopK} onChange={(e) => setEvidenceTopK(parseInt(e.target.value, 10))} /></label>
          <label>temperature <input type="number" step={0.05} value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} /></label>
        </div>
        <div>
          <strong>Answer</strong>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{answer}</pre>
          <strong>Citations</strong>
          <ul>
            {citations.map((c, i) => (
              <li key={i}>{c.doc_id} — pgs {c.pages} — {c.heading} — score {c.score?.toFixed?.(3)}</li>
            ))}
          </ul>
          <strong>Meta</strong>
          <pre>{JSON.stringify(meta, null, 2)}</pre>
        </div>
      </section>
    </div>
  )
}
