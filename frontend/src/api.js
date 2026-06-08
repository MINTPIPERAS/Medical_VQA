/**
 * Medical VQA API 客户端
 * 直连后端 localhost:7860，绕过 Vite 代理
 */

const BACKEND = 'http://localhost:7860'

export async function checkHealth() {
  const res = await fetch(`${BACKEND}/api/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
  return res.json()
}

export function streamVQA(imageFile, question, { onToken, onDone, onError }) {
  const controller = new AbortController()
  const formData = new FormData()
  formData.append('image', imageFile)
  formData.append('question', question || '请仔细观察这张皮肤病变图像，给出最可能的诊断及诊断依据。')
  formData.append('stream', 'true')

  console.log('[api] 发起请求到:', `${BACKEND}/api/vqa`)

  fetch(`${BACKEND}/api/vqa`, {
    method: 'POST',
    body: formData,
    signal: controller.signal,
  })
    .then(async (response) => {
      console.log('[api] 收到响应:', response.status, response.headers.get('content-type'))

      if (!response.ok) {
        const errText = await response.text()
        throw new Error(errText || `Server error: ${response.status}`)
      }

      if (!response.body) {
        throw new Error('浏览器不支持 ReadableStream 或 CORS 阻止了响应体读取')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finished = false
      let chunkCount = 0

      console.log('[api] 开始读取流...')

      while (true) {
        const { done, value } = await reader.read()

        if (value) {
          chunkCount++
          const chunk = decoder.decode(value, { stream: true })
          console.log(`[api] 收到 chunk #${chunkCount}:`, chunk.substring(0, 80))

          buffer += chunk
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (processLine(line, onToken, onDone, onError)) {
              finished = true
            }
          }
        }

        if (done) {
          console.log('[api] 流结束, finished=', finished)
          if (buffer.trim()) {
            processLine(buffer, onToken, onDone, onError)
          }
          if (!finished) {
            onDone && onDone()
          }
          break
        }
      }
    })
    .catch((err) => {
      console.error('[api] 错误:', err)
      if (err.name === 'AbortError') return
      onError && onError(err.message || String(err))
    })

  return controller
}

function processLine(line, onToken, onDone, onError) {
  const clean = line.replace(/\r$/, '')
  if (!clean.startsWith('data: ')) return false

  const data = clean.slice(6)
  if (data === '[DONE]') {
    console.log('[api] 收到 [DONE]')
    onDone && onDone()
    return true
  }
  if (data.startsWith('[ERROR]')) {
    console.log('[api] 收到错误:', data)
    onError && onError(data.slice(8))
    return true
  }
  onToken && onToken(data)
  return false
}
