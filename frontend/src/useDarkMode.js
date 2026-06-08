import { ref } from 'vue'

// 全局单例：暗色模式状态
const isDark = ref(false)

// 获取初始状态：localStorage > 系统偏好
function getInitialDark() {
  const stored = localStorage.getItem('theme')
  if (stored === 'dark') return true
  if (stored === 'light') return false
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

export function useDarkMode() {
  // 仅首次调用时初始化
  if (!window.__darkModeInited) {
    window.__darkModeInited = true
    isDark.value = getInitialDark()
    applyDark(isDark.value)

    // 监听系统偏好变化（仅在用户未手动设定时生效）
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (localStorage.getItem('theme') === null) {
        isDark.value = e.matches
        applyDark(e.matches)
      }
    })
  }

  function toggle() {
    isDark.value = !isDark.value
    applyDark(isDark.value)
    localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
  }

  return { isDark, toggle }
}

function applyDark(dark) {
  document.documentElement.classList.toggle('dark', dark)
}
