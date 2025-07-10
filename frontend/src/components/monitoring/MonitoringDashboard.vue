<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <v-card>
          <v-card-title>システム監視ダッシュボード</v-card-title>
          <v-card-text>
            <v-row>
              <!-- ヘルスステータス -->
              <v-col cols="12" md="6">
                <v-card variant="outlined">
                  <v-card-title>システムヘルス</v-card-title>
                  <v-card-text>
                    <v-chip 
                      :color="healthStatus.status === 'healthy' ? 'success' : 'error'"
                      size="large"
                    >
                      {{ healthStatus.status || 'unknown' }}
                    </v-chip>
                    <div class="mt-2">
                      <small>稼働時間: {{ formatUptime(healthStatus.uptime) }}</small>
                    </div>
                    <div class="mt-2">
                      <small>最終更新: {{ formatTime(healthStatus.timestamp) }}</small>
                    </div>
                  </v-card-text>
                </v-card>
              </v-col>

              <!-- メトリクス -->
              <v-col cols="12" md="6">
                <v-card variant="outlined">
                  <v-card-title>アプリケーションメトリクス</v-card-title>
                  <v-card-text>
                    <div class="d-flex justify-space-between">
                      <span>リクエスト数:</span>
                      <strong>{{ metrics.application?.request_count || 0 }}</strong>
                    </div>
                    <div class="d-flex justify-space-between">
                      <span>エラー数:</span>
                      <strong>{{ metrics.application?.error_count || 0 }}</strong>
                    </div>
                    <div class="d-flex justify-space-between">
                      <span>エラー率:</span>
                      <strong>{{ formatPercent(metrics.application?.error_rate) }}%</strong>
                    </div>
                    <div class="d-flex justify-space-between">
                      <span>平均応答時間:</span>
                      <strong>{{ metrics.application?.average_response_time || 0 }}ms</strong>
                    </div>
                  </v-card-text>
                </v-card>
              </v-col>

              <!-- システムリソース -->
              <v-col cols="12" md="6">
                <v-card variant="outlined">
                  <v-card-title>システムリソース</v-card-title>
                  <v-card-text>
                    <div class="d-flex justify-space-between">
                      <span>ディスク使用率:</span>
                      <strong>{{ formatPercent(metrics.system?.disk_percent) }}%</strong>
                    </div>
                    <div class="d-flex justify-space-between">
                      <span>ディスク容量:</span>
                      <strong>{{ formatBytes(metrics.system?.disk_total) }}</strong>
                    </div>
                    <div class="d-flex justify-space-between">
                      <span>ディスク空き:</span>
                      <strong>{{ formatBytes(metrics.system?.disk_free) }}</strong>
                    </div>
                  </v-card-text>
                </v-card>
              </v-col>

              <!-- 詳細ヘルスチェック -->
              <v-col cols="12" md="6">
                <v-card variant="outlined">
                  <v-card-title>詳細チェック</v-card-title>
                  <v-card-text>
                    <div v-for="(check, key) in detailedHealth.checks" :key="key" class="d-flex justify-space-between">
                      <span>{{ key }}:</span>
                      <v-chip
                        :color="getCheckColor(check.status)"
                        size="small"
                      >
                        {{ check.status }}
                      </v-chip>
                    </div>
                  </v-card-text>
                </v-card>
              </v-col>
            </v-row>

            <!-- 最新ログ -->
            <v-row class="mt-4">
              <v-col cols="12">
                <v-card variant="outlined">
                  <v-card-title>
                    最新ログ
                    <v-spacer></v-spacer>
                    <v-btn @click="scrollToTop" size="small" variant="outlined" class="mr-2">
                      上へ
                    </v-btn>
                    <v-btn @click="scrollToBottom" size="small" variant="outlined" class="mr-2">
                      下へ
                    </v-btn>
                    <v-btn @click="clearLogs" size="small" variant="outlined" class="mr-2" color="warning">
                      クリア
                    </v-btn>
                    <v-btn @click="fetchLogs" size="small" variant="outlined">更新</v-btn>
                  </v-card-title>
                  <v-card-text>
                    <div 
                      class="log-container"
                    >
                      <div 
                        v-for="(log, index) in recentLogs" 
                        :key="index"
                        class="text-caption font-mono mb-1 log-entry"
                      >
                        {{ log }}
                      </div>
                      <div v-if="recentLogs.length === 0" class="text-center text-grey">
                        ログがありません
                      </div>
                    </div>
                  </v-card-text>
                </v-card>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import axios from 'axios'

const healthStatus = ref({})
const detailedHealth = ref({ checks: {} })
const metrics = ref({ application: {}, system: {} })
const recentLogs = ref([])
const refreshInterval = ref(null)

async function fetchHealthStatus() {
  try {
    const response = await axios.get('/api/monitoring/health')
    healthStatus.value = response.data
  } catch (error) {
    console.error('ヘルスステータス取得エラー:', error)
    healthStatus.value = { status: 'error' }
  }
}

async function fetchDetailedHealth() {
  try {
    const response = await axios.get('/api/monitoring/health/detailed')
    detailedHealth.value = response.data
  } catch (error) {
    console.error('詳細ヘルスチェック取得エラー:', error)
    detailedHealth.value = { checks: {} }
  }
}

async function fetchMetrics() {
  try {
    const response = await axios.get('/api/monitoring/metrics')
    metrics.value = response.data
  } catch (error) {
    console.error('メトリクス取得エラー:', error)
    metrics.value = { application: {}, system: {} }
  }
}

async function fetchLogs() {
  try {
    const response = await axios.get('/api/monitoring/logs?lines=50')
    recentLogs.value = response.data.logs || []
    
    // 新しいログが追加された場合、自動的に最下部にスクロール
    await nextTick()
    const logContainer = document.querySelector('.log-container')
    if (logContainer) {
      logContainer.scrollTop = logContainer.scrollHeight
    }
  } catch (error) {
    console.error('ログ取得エラー:', error)
    recentLogs.value = []
  }
}

async function fetchAllData() {
  await Promise.all([
    fetchHealthStatus(),
    fetchDetailedHealth(),
    fetchMetrics(),
    fetchLogs()
  ])
}

function formatUptime(uptime) {
  if (!uptime) return 'N/A'
  const hours = Math.floor(uptime / 3600)
  const minutes = Math.floor((uptime % 3600) / 60)
  const seconds = Math.floor(uptime % 60)
  return `${hours}h ${minutes}m ${seconds}s`
}

function formatTime(timestamp) {
  if (!timestamp) return 'N/A'
  return new Date(timestamp).toLocaleString()
}

function formatPercent(value) {
  if (value === undefined || value === null) return 'N/A'
  return Math.round(value * 100) / 100
}

function formatBytes(bytes) {
  if (!bytes) return 'N/A'
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i]
}

function getCheckColor(status) {
  switch (status) {
    case 'healthy': return 'success'
    case 'degraded': return 'warning'
    case 'unhealthy': return 'error'
    default: return 'grey'
  }
}

function scrollToTop() {
  const logContainer = document.querySelector('.log-container')
  if (logContainer) {
    logContainer.scrollTop = 0
  }
}

function scrollToBottom() {
  const logContainer = document.querySelector('.log-container')
  if (logContainer) {
    logContainer.scrollTop = logContainer.scrollHeight
  }
}

function clearLogs() {
  recentLogs.value = []
}

onMounted(() => {
  fetchAllData()
  // 30秒ごとに更新
  refreshInterval.value = setInterval(fetchAllData, 30000)
})

onUnmounted(() => {
  if (refreshInterval.value) {
    clearInterval(refreshInterval.value)
  }
})
</script>

<style scoped>
.font-mono {
  font-family: 'Courier New', monospace;
}

.log-container {
  max-height: 300px;
  overflow-y: auto;
  background-color: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 8px;
}

.log-entry {
  word-break: break-all;
  white-space: pre-wrap;
  border-bottom: 1px solid #eee;
  padding: 4px 0;
}

.log-entry:last-child {
  border-bottom: none;
}

/* ダークテーマ対応 */
.v-theme--dark .log-container {
  background-color: #2d2d2d;
  border-color: #555;
}

.v-theme--dark .log-entry {
  border-bottom-color: #444;
}

/* スクロールバーのスタイリング */
.log-container::-webkit-scrollbar {
  width: 8px;
}

.log-container::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.log-container::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

.log-container::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

.v-theme--dark .log-container::-webkit-scrollbar-track {
  background: #444;
}

.v-theme--dark .log-container::-webkit-scrollbar-thumb {
  background: #666;
}

.v-theme--dark .log-container::-webkit-scrollbar-thumb:hover {
  background: #777;
}
</style>
