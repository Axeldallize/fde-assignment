import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ingestFiles, queryApi } from './api'

export default function App() {
  const [files, setFiles] = useState<File[] | null>(null)
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
  // Work around TS type mismatch for react-markdown in some toolchains
  const Markdown: any = ReactMarkdown

  const labelStyle: React.CSSProperties = { fontSize: 12, color: '#334155', fontWeight: 600 }
  const inputStyle: React.CSSProperties = { width: '90%', padding: 6, border: '1px solid #cbd5e1', borderRadius: 4, fontSize: 14 }
  const checkboxStyle: React.CSSProperties = { accentColor: '#2563eb', width: 18, height: 18 }

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const f = Array.from(e.dataTransfer.files || []).filter((x) => x.type === 'application/pdf')
    setFiles(f)
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
      <h2 style={{ marginBottom: 8 }}>RAG UI</h2>
      <p style={{ color: '#666', marginTop: 0 }}>Upload PDFs and ask questions. Toggle retrieval knobs below.</p>

      <section style={{ marginBottom: 24, padding: 16, border: '1px solid #e5e7eb', borderRadius: 12, background: '#fafafa' }}>
        <h3>Upload PDFs</h3>
        <div
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
          style={{ padding: 24, border: '2px dashed #cbd5e1', borderRadius: 8, marginBottom: 12, background: '#fff' }}
        >
          Drag & drop files here
        </div>
        <input type="file" multiple accept="application/pdf" onChange={(e) => setFiles(e.target.files ? Array.from(e.target.files) : null)} />
        <button onClick={onUpload} disabled={busy || !files || files.length === 0} style={{ marginLeft: 12, padding: '6px 12px' }}>Ingest</button>
      </section>

      <section style={{ marginBottom: 24, padding: 16, border: '1px solid #e5e7eb', borderRadius: 12, background: '#fafafa' }}>
        <h3>Ask</h3>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask a question" style={{ flex: 1, padding: 8, border: '1px solid #cbd5e1', borderRadius: 6 }} />
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="auto">auto</option>
            <option value="qa">qa</option>
            <option value="list">list</option>
            <option value="table">table</option>
          </select>
          <button onClick={onAsk} disabled={busy} style={{ padding: '6px 12px' }}>Send</button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 12, alignItems: 'end' }}>
          <div>
            <div style={labelStyle}>top_k</div>
            <input type="number" value={topK} min={1} onChange={(e) => setTopK(parseInt(e.target.value, 10) || 1)} style={inputStyle} />
          </div>
          <div>
            <div style={labelStyle}>semantic</div>
            <input type="checkbox" checked={useSemantic} onChange={(e) => setUseSemantic(e.target.checked)} style={checkboxStyle} />
          </div>
          <div>
            <div style={labelStyle}>use_rrf</div>
            <input type="checkbox" checked={useRrf} onChange={(e) => setUseRrf(e.target.checked)} style={checkboxStyle} />
          </div>
          <div>
            <div style={labelStyle}>threshold</div>
            <input type="number" step={0.01} value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value) || 0)} style={inputStyle} />
          </div>
          <div>
            <div style={labelStyle}>evidence_topk</div>
            <input type="number" min={1} value={evidenceTopK} onChange={(e) => setEvidenceTopK(parseInt(e.target.value, 10) || 1)} style={inputStyle} />
          </div>
          <div>
            <div style={labelStyle}>temperature</div>
            <input type="number" step={0.05} value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)} style={inputStyle} />
          </div>
        </div>
        <div>
          <strong>Answer</strong>
          <div style={{ padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
            <Markdown remarkPlugins={[remarkGfm]}>{answer || ''}</Markdown>
          </div>
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
