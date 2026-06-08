<script setup>
import { ref } from 'vue'

const props = defineProps({
  message: { type: Object, required: true },
})

const copied = ref(false)

async function copyContent() {
  try {
    await navigator.clipboard.writeText(props.message.content)
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  } catch {
    // fallback
    const ta = document.createElement('textarea')
    ta.value = props.message.content
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    document.body.removeChild(ta)
  }
}

function simpleMarkdown(text) {
  if (!text) return ''
  // 粗体
  let html = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  // 换行
  html = html.replace(/\n/g, '<br/>')
  return html
}
</script>

<template>
  <div
    :class="[
      'flex flex-col gap-1.5 group',
      message.role === 'user' ? 'items-end' : 'items-start',
    ]"
  >
    <!-- 用户上传的图片：独立于气泡，显示在气泡上方 -->
    <img
      v-if="message.role === 'user' && message.imageDataUrl"
      :src="message.imageDataUrl"
      class="max-w-[60%] max-h-56 rounded-xl object-cover shadow-md border border-neutral-200/60 dark:border-neutral-700/50"
      alt="上传的图像"
    />

    <!-- 头像 + 气泡行 -->
    <div
      :class="[
        'flex gap-3',
        message.role === 'user' ? 'justify-end' : 'justify-start',
      ]"
    >
      <!-- AI 头像 -->
      <div v-if="message.role === 'assistant'" class="shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-xs font-semibold shadow-sm">
        AI
      </div>

      <!-- 消息气泡 -->
      <div
        :class="[
          'relative max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm',
          message.role === 'user'
            ? 'bg-[--color-bubble-user] dark:bg-[--color-bubble-user-dark] text-gray-900 dark:text-white rounded-br-md border border-blue-300/60 dark:border-white/50 dark:shadow-md'
            : 'bg-neutral-100 dark:bg-neutral-800 text-gray-900 dark:text-gray-100 rounded-bl-md border border-neutral-200 dark:border-neutral-600/40',
          message.error ? 'border border-red-400 text-red-600 dark:text-red-400' : '',
        ]"
      >
        <!-- 消息内容 -->
        <div
          v-if="message.content"
          class="message-content"
          v-html="simpleMarkdown(message.content)"
        />

        <!-- 等待首 token 时的提示（CPU 模式下可能较久） -->
        <div
          v-if="message.streaming && !message.content"
          class="flex items-center gap-2 text-gray-400 dark:text-gray-500"
        >
          <span class="inline-flex gap-0.5">
            <span class="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0ms"></span>
            <span class="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 150ms"></span>
            <span class="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 300ms"></span>
          </span>
          <span class="text-xs text-gray-400 dark:text-gray-500">正在分析图像，请稍候…</span>
        </div>

        <!-- 加载动画 -->
        <span
          v-if="message.streaming && message.content"
          class="inline-block w-2 h-4 bg-blue-400 dark:bg-blue-300 animate-pulse rounded-sm ml-0.5 align-middle"
        />

        <!-- 操作栏：复制按钮 -->
        <div
          v-if="message.role === 'assistant' && message.content && !message.streaming"
          class="flex items-center gap-1 mt-1.5 pt-1 border-t border-neutral-200 dark:border-neutral-700"
        >
          <button
            @click="copyContent"
            class="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
            :title="copied ? '已复制' : '复制回答'"
          >
            <svg v-if="!copied" class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <svg v-else class="w-3.5 h-3.5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
            <span>{{ copied ? '已复制' : '复制' }}</span>
          </button>
        </div>
      </div>

      <!-- 用户头像 -->
      <div v-if="message.role === 'user'" class="shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center text-white text-xs font-semibold shadow-sm">
        U
      </div>
    </div>
  </div>
</template>
