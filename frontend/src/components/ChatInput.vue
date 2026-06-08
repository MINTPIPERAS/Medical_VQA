<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  currentImage: { type: Object, default: null },
  isLoading: { type: Boolean, default: false },
})

const emit = defineEmits(['send'])

const question = ref('')
const imageFile = ref(null)
const imageDataUrl = ref(null)
const imageInputRef = ref(null)
const skipNextSync = ref(false)

// 清空本地图片状态（同时重置 file input）
function clearLocalImage() {
  imageFile.value = null
  imageDataUrl.value = null
  if (imageInputRef.value) {
    imageInputRef.value.value = ''
  }
}

// 同步外部图片状态（仅当有新图片时载入，新建对话时清空）
watch(() => props.currentImage, (val) => {
  // 刚发送完消息时，父组件会更新 currentImage 以保留图片供多轮使用，
  // 但输入区应保持清空 — 通过 skipNextSync 跳过本次同步
  if (skipNextSync.value) {
    skipNextSync.value = false
    return
  }

  if (!val || !val.dataUrl) {
    // 新建对话 / 图片被清空 → 同步清空本地
    clearLocalImage()
  } else if (val.dataUrl) {
    // 从外部恢复（切换历史对话、首次加载）：同步 dataUrl；file 可能为 null（来自 DB）
    imageDataUrl.value = val.dataUrl
    imageFile.value = val.file || null
  }
}, { immediate: true })

function handleImageUpload(e) {
  const file = e.target.files?.[0]
  if (!file) return

  // 验证类型
  if (!file.type.startsWith('image/')) {
    alert('请选择图片文件（JPEG / PNG）')
    return
  }

  imageFile.value = file
  const reader = new FileReader()
  reader.onload = (ev) => {
    imageDataUrl.value = ev.target.result
  }
  reader.readAsDataURL(file)
}

function removeImage() {
  clearLocalImage()
}

function handleSend() {
  const q = question.value.trim()
  if (!q && !imageDataUrl.value) return

  // 先标记"跳过下次同步"再 emit — emit 会同步触发父组件的 handleSend，
  // 父组件可能更新 currentImage 进而触发本组件的 watch，标记位让它跳过回填
  skipNextSync.value = true

  emit('send', {
    question: q,
    imageFile: imageFile.value,
    imageDataUrl: imageDataUrl.value,
  })

  // 发送后清空输入内容与本地图片预览
  question.value = ''
  clearLocalImage()

  // 在下一轮微任务中复位标记位，避免漏网影响后续正常的 watch 同步
  nextTick(() => {
    skipNextSync.value = false
  })
}
</script>

<template>
  <div class="border-t border-neutral-200/60 dark:border-neutral-800/60 bg-white/80 dark:bg-neutral-900/80 backdrop-blur-sm px-4 py-3">
    <div class="max-w-3xl mx-auto">
      <!-- 图片预览 -->
      <div v-if="imageDataUrl" class="mb-2 relative inline-block">
        <img
          :src="imageDataUrl"
          class="h-20 rounded-lg object-cover border border-neutral-200 dark:border-neutral-700"
          alt="Preview"
        />
        <button
          @click="removeImage"
          class="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-red-500 text-white
                 flex items-center justify-center text-xs hover:bg-red-600 transition-colors"
          title="移除图片"
        >
          ×
        </button>
      </div>

      <!-- 输入栏 -->
      <div class="flex items-end gap-1.5 bg-neutral-100 dark:bg-neutral-800 rounded-2xl px-2 py-1.5
                  border border-transparent focus-within:border-blue-400 dark:focus-within:border-blue-500
                  focus-within:bg-white dark:focus-within:bg-neutral-800
                  shadow-sm dark:shadow-none
                  transition-all duration-200">
        <!-- 文本输入 -->
        <textarea
          v-model="question"
          @keydown.enter.exact.prevent="handleSend"
          placeholder="输入您的问题，如：请分析这张皮肤病变图像..."
          rows="1"
          class="flex-1 resize-none bg-transparent px-2 py-1.5 text-sm
                 placeholder-gray-400 dark:placeholder-gray-500
                 text-gray-900 dark:text-gray-100
                 focus:outline-none
                 transition-colors"
          :disabled="isLoading"
        />

        <!-- 回形针图片上传按钮 -->
        <label
          class="shrink-0 p-1.5 rounded-full cursor-pointer
                 text-gray-400 hover:text-blue-500 dark:hover:text-blue-400
                 hover:bg-blue-50 dark:hover:bg-blue-500/10
                 transition-all duration-200"
          title="上传图片"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
          </svg>
          <input
            ref="imageInputRef"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            class="hidden"
            @change="handleImageUpload"
          />
        </label>

        <!-- 发送按钮 -->
        <button
          @click="handleSend"
          :disabled="isLoading || (!question.trim() && !imageDataUrl)"
          class="shrink-0 p-1.5 rounded-full transition-all duration-200
                 bg-blue-500 text-white
                 hover:bg-blue-600 hover:shadow-md
                 active:scale-95
                 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:shadow-none"
          title="发送"
        >
          <svg v-if="!isLoading" class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
          <svg v-else class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </button>
      </div>

      <!-- 提示文字 -->
      <div class="text-[11px] text-gray-400 dark:text-gray-500 mt-2 text-center select-none">
        上传病变图像后输入问题，按 Enter 发送 · AI 回答仅供参考，不可替代医生诊断
      </div>
    </div>
  </div>
</template>
