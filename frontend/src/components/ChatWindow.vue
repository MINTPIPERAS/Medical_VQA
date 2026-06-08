<script setup>
import { ref, nextTick, watch } from 'vue'
import MessageBubble from './MessageBubble.vue'
import ChatInput from './ChatInput.vue'
import { useDarkMode } from '../useDarkMode.js'

const { isDark, toggle: toggleDark } = useDarkMode()

const props = defineProps({
  messages: { type: Array, default: () => [] },
  currentImage: { type: Object, default: null },
  isLoading: { type: Boolean, default: false },
  sidebarOpen: { type: Boolean, default: true },
})

const emit = defineEmits(['send', 'toggle-sidebar'])

const chatContainer = ref(null)

// 新消息到达时自动滚到底部
watch(
  () => props.messages.length,
  async () => {
    await nextTick()
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  },
)

// 流式内容更新时也滚动
watch(
  () => {
    const msgs = props.messages
    if (msgs.length === 0) return ''
    const last = msgs[msgs.length - 1]
    return last.streaming ? last.content : ''
  },
  async () => {
    await nextTick()
    if (chatContainer.value) {
      const el = chatContainer.value
      // 只有在接近底部时才自动滚动（避免打断用户回看）
      if (el.scrollHeight - el.scrollTop - el.clientHeight < 100) {
        el.scrollTop = el.scrollHeight
      }
    }
  },
)
</script>

<template>
  <div class="flex-1 flex flex-col min-w-0 h-full">
    <!-- 顶部栏 -->
    <header class="flex items-center gap-3 px-4 h-14 border-b border-neutral-200/60 dark:border-neutral-800/60 bg-white/80 dark:bg-neutral-900/80 backdrop-blur-sm">
      <button
        v-if="!sidebarOpen"
        @click="emit('toggle-sidebar')"
        class="p-1.5 rounded-full hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
        title="打开侧边栏"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      <div class="text-sm font-medium text-gray-600 dark:text-gray-400 select-none">
        皮肤病变视觉问答系统
      </div>
      <div class="flex-1" />
      <!-- 暗色模式切换 -->
      <button
        @click="toggleDark"
        class="p-1.5 rounded-full hover:bg-neutral-100 dark:hover:bg-neutral-800 text-gray-500 transition-colors"
        title="切换主题"
      >
        <!-- 暗色模式下显示太阳（切回亮色） -->
        <svg v-if="isDark" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
        <!-- 亮色模式下显示月亮（切到暗色） -->
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
        </svg>
      </button>
    </header>

    <!-- 消息区域 -->
    <div
      ref="chatContainer"
      class="flex-1 overflow-y-auto px-4 py-6 bg-neutral-50/50 dark:bg-neutral-950/50"
    >
      <!-- 空状态 -->
      <div v-if="messages.length === 0" class="flex flex-col items-center justify-center h-full">
        <div class="w-20 h-20 rounded-full bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-500/10 dark:to-purple-500/10 flex items-center justify-center mb-6 shadow-sm">
          <span class="text-4xl">🩺</span>
        </div>
        <div class="text-xl font-semibold text-gray-700 dark:text-gray-200 mb-2">
          Medical VQA
        </div>
        <div class="text-sm text-gray-400 dark:text-gray-500 max-w-md text-center leading-relaxed mb-2">
          皮肤病变视觉问答系统
        </div>
        <div class="text-sm text-gray-400 dark:text-gray-500 max-w-md text-center leading-relaxed">
          上传一张皮肤病变图像，并输入您的问题。<br/>
          系统将基于 AI 模型给出专业的医学分析。
        </div>
        <div class="mt-6 flex flex-wrap items-center justify-center gap-2">
          <span class="px-2.5 py-1 text-xs rounded-full bg-blue-50 text-blue-600 dark:bg-blue-500/10 dark:text-blue-300">水痘</span>
          <span class="px-2.5 py-1 text-xs rounded-full bg-purple-50 text-purple-600 dark:bg-purple-500/10 dark:text-purple-300">猴痘</span>
          <span class="px-2.5 py-1 text-xs rounded-full bg-orange-50 text-orange-600 dark:bg-orange-500/10 dark:text-orange-300">手足口病</span>
          <span class="px-2.5 py-1 text-xs rounded-full bg-red-50 text-red-600 dark:bg-red-500/10 dark:text-red-300">麻疹</span>
          <span class="px-2.5 py-1 text-xs rounded-full bg-teal-50 text-teal-600 dark:bg-teal-500/10 dark:text-teal-300">牛痘</span>
          <span class="px-2.5 py-1 text-xs rounded-full bg-green-50 text-green-600 dark:bg-green-500/10 dark:text-green-300">健康皮肤</span>
        </div>
      </div>

      <!-- 消息列表 -->
      <div v-else class="max-w-3xl mx-auto space-y-4">
        <MessageBubble
          v-for="(msg, idx) in messages"
          :key="idx"
          :message="msg"
        />
      </div>
    </div>

    <!-- 输入区域 -->
    <ChatInput
      :currentImage="currentImage"
      :isLoading="isLoading"
      @send="(data) => emit('send', data)"
    />
  </div>
</template>
