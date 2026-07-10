import { useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

const API_BASE = 'http://127.0.0.1:8000'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const eventSourceRef = useRef(null)
  const [error, setError] = useState(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    const question = input.trim()
    if (!question || isStreaming) return

    // Add the user's question, and a placeholder assistant message we'll
    // fill in as tokens stream in.
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: question },
      { role: 'assistant', content: '', citations: null },
    ])
    setInput('')
    setIsStreaming(true)
    setError(null)

    const url = `${API_BASE}/query/stream?question=${encodeURIComponent(question)}&k=5`
    const es = new EventSource(url)
    eventSourceRef.current = es

    es.addEventListener('token', (event) => {
      const data = JSON.parse(event.data)
      setMessages((prev) => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        updated[updated.length - 1] = { ...last, content: last.content + data.token }
        return updated
      })
    })

    es.addEventListener('citations', (event) => {
      const data = JSON.parse(event.data)
      setMessages((prev) => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        updated[updated.length - 1] = { ...last, citations: data.citations }
        return updated
      })
      es.close()
      setIsStreaming(false)
    })

    es.onerror = () => {
      es.close()
      setIsStreaming(false)
      // Only show an error if we never got a proper close via the
      // citations event — EventSource sometimes fires onerror even
      // after a clean finish, so guard against a false-positive banner.
      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (last && last.role === 'assistant' && last.citations === null) {
          setError('Something went wrong while generating the answer. Please try again.')
        }
        return prev
      })
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 flex flex-col">
      <header className="border-b border-slate-700 p-4">
        <h1 className="text-xl font-semibold text-white">Corvex</h1>
        <p className="text-sm text-slate-400">Ask questions about the requests codebase</p>
      </header>

      <main className="flex-1 overflow-y-auto p-4 space-y-4 max-w-3xl mx-auto w-full">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role === 'user' ? 'text-right' : 'text-left'}>
            <div
              className={
                msg.role === 'user'
                  ? 'inline-block bg-blue-600 text-white rounded-lg px-4 py-2 max-w-xl text-left'
                  : 'inline-block bg-slate-800 text-slate-100 rounded-lg px-4 py-2 max-w-2xl text-left'
              }
            >
              {msg.role === 'assistant' ? (
                <>
                  <ReactMarkdown
                    components={{
                      code({ inline, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || '')
                        return !inline && match ? (
                          <SyntaxHighlighter
                            style={oneDark}
                            language={match[1]}
                            PreTag="div"
                            {...props}
                          >
                            {String(children).replace(/\n$/, '')}
                          </SyntaxHighlighter>
                        ) : (
                          <code className="bg-slate-700 px-1 py-0.5 rounded text-sm" {...props}>
                            {children}
                          </code>
                        )
                      },
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                  {msg.citations && (
                    <div className="mt-3 pt-3 border-t border-slate-700 space-y-1">
                      {msg.citations.length > 0 ? (
                        <>
                          <p className="text-xs text-slate-400 font-semibold">Sources:</p>
                          {msg.citations.map((c, ci) => (
                            <div key={ci} className="text-xs text-slate-400">
                              <span className="text-blue-400">{c.symbol_name || '(module-level)'}</span>
                              {' — '}
                              <span className="font-mono">{c.filepath.replace('../corvex_data/requests/src/', '')}</span>
                              {c.start_line && c.end_line && (
                                <span className="text-slate-500"> (lines {c.start_line}–{c.end_line})</span>
                              )}
                            </div>
                          ))}
                        </>
                      ) : (
                        <p className="text-xs text-slate-500 italic">No sources cited for this response.</p>
                      )}
                    </div>
                  )}
                </>
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}
        {isStreaming && messages[messages.length - 1]?.content === '' && (
          <div className="text-left">
            <div className="inline-block bg-slate-800 text-slate-400 rounded-lg px-4 py-2 text-sm italic">
              Thinking...
            </div>
          </div>
        )}
      </main>

      {error && (
        <div className="max-w-3xl mx-auto w-full px-4 pb-2">
          <div className="bg-red-900/50 border border-red-700 text-red-200 text-sm rounded-lg px-4 py-2">
            {error}
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="border-t border-slate-700 p-4 flex gap-2 max-w-3xl mx-auto w-full">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the codebase..."
          className="flex-1 bg-slate-800 text-white rounded-lg px-4 py-2 outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isStreaming}
        />
        <button
          type="submit"
          disabled={isStreaming || !input.trim()}
          className="bg-blue-600 text-white rounded-lg px-6 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isStreaming ? '...' : 'Ask'}
        </button>
      </form>
    </div>
  )
}

export default App