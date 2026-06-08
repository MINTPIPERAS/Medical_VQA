<script setup>
defineProps({
  conversations: { type: Array, default: () => [] },
  currentId: { type: Number, default: null },
  isOpen: { type: Boolean, default: true },
})

const emit = defineEmits(['new-chat', 'select', 'delete', 'toggle'])

function formatDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  const now = new Date()
  const diff = now - d
  if (diff < 86400000) {
    return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}
</script>

<template>
  <!-- 遮罩层（移动端） -->
  <div
    v-if="isOpen"
    class="fixed inset-0 bg-black/50 z-20 lg:hidden"
    @click="emit('toggle')"
  />

  <!-- 侧边栏 -->
  <aside
    :class="[
      'fixed lg:relative z-30 h-full flex flex-col transition-transform duration-300',
      'bg-neutral-50 dark:bg-neutral-950 border-r border-neutral-200 dark:border-neutral-800',
      'w-72',
      isOpen ? 'translate-x-0' : '-translate-x-full lg:hidden',
    ]"
  >
    <!-- 头部 -->
    <div class="relative flex items-center justify-center px-4 h-14 border-b border-neutral-200/60 dark:border-neutral-800/60">
      <h1 class="text-base font-semibold truncate text-gray-800 dark:text-gray-200">🩺 Medical VQA</h1>
      <button
        @click="emit('toggle')"
        class="lg:hidden absolute right-3 p-1.5 rounded-full hover:bg-neutral-200 dark:hover:bg-neutral-800 transition-colors"
        title="关闭侧边栏"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <!-- 新建对话按钮 -->
    <div class="px-3 py-2">
      <button
        @click="emit('new-chat')"
        class="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm rounded-full
               bg-blue-500 hover:bg-blue-600 text-white
               dark:bg-blue-700 dark:hover:bg-blue-600
               shadow-sm hover:shadow-md
               active:scale-[0.97]
               transition-all duration-200"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 4v16m8-8H4" />
        </svg>
        新建对话
      </button>
    </div>

    <!-- 对话列表 -->
    <div class="flex-1 overflow-y-auto px-2 py-1">
      <div v-if="conversations.length === 0" class="text-center text-xs text-gray-400 dark:text-gray-500 mt-8 px-4 leading-relaxed">
        暂无对话记录<br/>上传医学图像开始诊断
      </div>
      <div
        v-for="conv in conversations"
        :key="conv.id"
        @click="emit('select', conv.id)"
        :class="[
          'group flex items-center gap-2 px-3 py-2.5 my-0.5 rounded-xl cursor-pointer transition-all duration-200',
          currentId === conv.id
            ? 'bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-300'
            : 'hover:bg-neutral-100 dark:hover:bg-neutral-800/60 text-gray-700 dark:text-gray-300',
        ]"
      >
        <div class="flex-1 min-w-0">
          <div class="text-sm truncate">{{ conv.title }}</div>
          <div class="text-xs text-gray-400 dark:text-gray-500 mt-0.5">{{ formatDate(conv.updatedAt) }}</div>
        </div>
        <button
          @click.stop="emit('delete', conv.id)"
          class="opacity-0 group-hover:opacity-100 p-1 rounded-full hover:bg-red-100 dark:hover:bg-red-500/10 text-gray-400 hover:text-red-500 transition-all duration-200"
          title="删除对话"
        >
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>
    </div>
  </aside>
</template>
