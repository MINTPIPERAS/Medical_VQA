<script setup>
import { ref, onMounted } from 'vue'
import ChatSidebar from './components/ChatSidebar.vue'
import ChatWindow from './components/ChatWindow.vue'
import { getConversationList, getConversation, saveConversation, deleteConversation, generateTitle } from './db.js'

// 当前对话数据
const conversations = ref([])
const currentId = ref(null)
const messages = ref([])
const currentImage = ref(null)      // { file, dataUrl }
const isLoading = ref(false)
const sidebarOpen = ref(true)

// 初始化：从 IndexedDB 加载对话列表
onMounted(async () => {
  try {
    conversations.value = await getConversationList()
  } catch (e) {
    console.error('Failed to load conversations:', e)
  }
})

// 新建对话
function newChat() {
  currentId.value = null
  messages.value = []
  currentImage.value = null
  isLoading.value = false
}

// 选择历史对话
async function selectConversation(id) {
  if (id == currentId.value) return
  try {
    const conv = await getConversation(id)
    if (conv) {
      currentId.value = conv.id
      messages.value = conv.messages || []
      currentImage.value = conv.imageDataUrl
        ? { dataUrl: conv.imageDataUrl, file: null }
        : null
    }
  } catch (e) {
    console.error('Failed to load conversation:', e)
  }
}

// 删除对话
async function handleDelete(id) {
  if (id == null) return
  try {
    await deleteConversation(id)
    conversations.value = conversations.value.filter(c => c.id != id)
    if (currentId.value == id) {
      newChat()
    }
  } catch (e) {
    console.error('Failed to delete conversation:', e)
  }
}

// 发送消息
async function handleSend({ question, imageFile, imageDataUrl }) {
  // 防止并发重复调用
  if (isLoading.value) return
  isLoading.value = true

  try {
    if (!imageFile && !currentImage.value?.file && !currentImage.value?.dataUrl) {
      // 给出明确提示而非静默吞掉
      const msg = messages.value.length === 0
        ? '请先上传一张皮肤病变图像再提问。'
        : '当前对话没有关联图像，请上传图像后重试。'
      messages.value.push({
        role: 'assistant',
        content: '⚠️ ' + msg,
        timestamp: new Date().toISOString(),
        error: true,
      })
      isLoading.value = false
      return
    }

    // 如果没有当前对话 ID，先创建
    if (currentId.value === null) {
      try {
        const id = await saveConversation({
          title: generateTitle(question),
          messages: [],
          imageDataUrl: imageDataUrl || currentImage.value?.dataUrl || null,
        })
        currentId.value = id
        conversations.value = await getConversationList()
      } catch (e) {
        console.error('Failed to create conversation:', e)
        const errMsg = e?.message || String(e)
        messages.value.push({
          role: 'assistant',
          content: '❌ 创建对话失败: ' + errMsg,
          timestamp: new Date().toISOString(),
          error: true,
        })
        isLoading.value = false
        return
      }
    }

    // 确定本次使用的图片 dataUrl（新上传优先，否则沿用已保存的）
    const imgDUrl = imageDataUrl || currentImage.value?.dataUrl || null

    // 只在真正有新图片文件且图片确实变化时才更新 currentImage，
    // 保护多轮对话的 file 引用，同时避免无变化时触发 ChatInput 的 watch
    if (imageFile) {
      if (!currentImage.value || currentImage.value.dataUrl !== imgDUrl) {
        currentImage.value = { file: imageFile, dataUrl: imgDUrl }
      }
    } else if (!currentImage.value && imgDUrl) {
      // 从历史恢复等场景：只有 dataUrl 没有 file 对象
      currentImage.value = { file: null, dataUrl: imgDUrl }
    }

    // 添加用户消息
    const userMsg = {
      role: 'user',
      content: question || '请分析这张图像。',
      imageDataUrl: imgDUrl,
      timestamp: new Date().toISOString(),
    }
    messages.value.push(userMsg)

    // 添加 AI 占位消息
    const aiMsg = {
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      streaming: true,
    }
    messages.value.push(aiMsg)

    // 调用后端 API
    const { streamVQA } = await import('./api.js')
    const fileToSend = imageFile || (currentImage.value?.file)
    // 如果只有 dataUrl 没有 file，构建一个 Blob
    let actualFile = fileToSend
    if (!actualFile && imgDUrl) {
      const resp = await fetch(imgDUrl)
      const blob = await resp.blob()
      actualFile = new File([blob], 'image.jpg', { type: blob.type || 'image/jpeg' })
    }

    // aiMsg push 后被 Vue 包装为 reactive Proxy，必须通过数组索引访问才能触发响应式更新
    streamVQA(actualFile, question, {
      onToken(token) {
        messages.value[messages.value.length - 1].content += token
      },
      onDone() {
        const msg = messages.value[messages.value.length - 1]
        msg.streaming = false
        isLoading.value = false
        persistConversation()
      },
      onError(err) {
        const msg = messages.value[messages.value.length - 1]
        msg.content = '❌ 错误: ' + err
        msg.streaming = false
        msg.error = true
        isLoading.value = false
        persistConversation()
      },
    })
  } catch (e) {
    // 兜底：捕获所有未预期的异常
    console.error('handleSend error:', e)
    messages.value.push({
      role: 'assistant',
      content: '❌ 发送失败: ' + (e.message || String(e)),
      timestamp: new Date().toISOString(),
      error: true,
    })
    isLoading.value = false
  }
}

// 持久化当前对话到 IndexedDB
async function persistConversation() {
  if (currentId.value === null) return
  try {
    await saveConversation({
      id: currentId.value,
      title: generateTitle(messages.value.find(m => m.role === 'user')?.content || ''),
      messages: messages.value,
      imageDataUrl: currentImage.value?.dataUrl || null,
    })
    conversations.value = await getConversationList()
  } catch (e) {
    console.error('Failed to persist conversation:', e)
  }
}
</script>

<template>
  <div class="flex h-full bg-white dark:bg-neutral-950 text-gray-900 dark:text-gray-100">
    <!-- 侧边栏 -->
    <ChatSidebar
      :conversations="conversations"
      :currentId="currentId"
      :isOpen="sidebarOpen"
      @new-chat="newChat"
      @select="selectConversation"
      @delete="handleDelete"
      @toggle="sidebarOpen = !sidebarOpen"
    />

    <!-- 主聊天区 -->
    <ChatWindow
      :messages="messages"
      :currentImage="currentImage"
      :isLoading="isLoading"
      :sidebarOpen="sidebarOpen"
      @send="handleSend"
      @toggle-sidebar="sidebarOpen = !sidebarOpen"
    />
  </div>
</template>
