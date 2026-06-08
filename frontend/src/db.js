/**
 * IndexedDB 对话存储
 * 数据库名: MedicalVQA
 * 表: conversations — 每条记录是一个完整的对话
 */

const DB_NAME = 'MedicalVQA'
const DB_VERSION = 1
const STORE_NAME = 'conversations'

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onupgradeneeded = (event) => {
      const db = event.target.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, {
          keyPath: 'id',
          autoIncrement: true,
        })
        store.createIndex('updatedAt', 'updatedAt', { unique: false })
        store.createIndex('title', 'title', { unique: false })
      }
    }

    request.onsuccess = () => {
      const db = request.result
      // 监听连接关闭（版本变更等），自动重连
      db.onclose = () => {
        console.warn('[IndexedDB] 连接意外关闭')
      }
      resolve(db)
    }
    request.onerror = () => reject(request.error || new Error('IndexedDB 打开失败'))
    request.onblocked = () => {
      console.warn('[IndexedDB] 数据库被阻塞，尝试关闭旧连接...')
      // 关闭当前请求，让用户知道需要刷新
      reject(new Error('数据库被其他标签页阻塞，请关闭所有标签页后刷新重试'))
    }
  })
}

/**
 * 获取所有对话列表（按更新时间倒序，不含消息内容）
 */
export async function getConversationList() {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const request = store.getAll()

    request.onsuccess = () => {
      const list = request.result
        .map(({ id, title, createdAt, updatedAt, messageCount }) => ({
          id,
          title: title || 'New Conversation',
          createdAt,
          updatedAt,
          messageCount: messageCount || 0,
        }))
        .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
      resolve(list)
    }
    request.onerror = () => reject(request.error)
  })
}

/**
 * 获取单个对话的完整数据
 * @param {number} id
 */
export async function getConversation(id) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const request = store.get(id)

    request.onsuccess = () => resolve(request.result || null)
    request.onerror = () => reject(request.error)
  })
}

/**
 * 保存/更新对话
 * @param {object} conversation — { id?, title, messages, imageDataUrl?, createdAt?, updatedAt? }
 * @returns {number} 对话 id
 */
export async function saveConversation(conversation) {
  const db = await openDB()
  const now = new Date().toISOString()

  // 深拷贝：Vue 响应式代理对象无法被 IndexedDB 的结构化克隆接收
  const plainMessages = (conversation.messages || []).map(m => ({
    role: m.role,
    content: m.content,
    imageDataUrl: m.imageDataUrl || null,
    timestamp: m.timestamp || now,
    streaming: false,  // 存储时永远为 false
    error: m.error || false,
  }))

  const record = {
    title: conversation.title || 'New Conversation',
    messages: plainMessages,
    imageDataUrl: conversation.imageDataUrl || null,
    updatedAt: now,
    messageCount: plainMessages.length,
    createdAt: conversation.createdAt || now,
  }
  // 新建对话时不带 id 字段，让 autoIncrement 自动生成；
  // 更新已有对话时显式传入 id 以确保覆盖而非新增
  if (conversation.id != null) {
    record.id = conversation.id
  }

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const request = store.put(record)

    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

/**
 * 删除对话
 * @param {number} id
 */
export async function deleteConversation(id) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const request = store.delete(id)

    request.onsuccess = () => resolve()
    request.onerror = () => reject(request.error)
  })
}

/**
 * 自动生成对话标题（取第一条用户问题的前30字）
 * @param {string} question
 */
export function generateTitle(question) {
  if (!question) return 'New Conversation'
  const cleaned = question.replace(/<image>/g, '').trim()
  return cleaned.length > 30 ? cleaned.slice(0, 30) + '...' : cleaned
}
