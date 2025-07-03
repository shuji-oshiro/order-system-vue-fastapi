<!-- メニュー画面の共通デザインカード -->
<template>

  <v-hover>
  <template v-slot:default="{ isHovering, props }">
    <v-card @click="onClick"
        v-bind="props"
      :color="isHovering ? 'primary' : undefined"
      elevated
      class="mx-auto"
      max-width="344px"
    >
      <!-- スロットでタイトル部分を外部から変更可能 -->
      <template #title>
        <slot name="menu-title">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>{{ menu.name }}</span>          
            <span style="font-weight: bold;">¥{{ menu.price }}</span>
          </div>
        </slot>
      </template>
      <div>
        <v-img
          height="300px"
          src="https://cdn.vuetifyjs.com/images/cards/sunshine.jpg"
          cover
        />
      </div>
      <template #actions>
        {{ menu.description }}
      </template>
    </v-card>
  </template>
</v-hover>

</template>

<script setup lang="ts">
  import type { MenuOut } from '@/types/menuTypes'

  defineProps<{
    menu: MenuOut
    onClick: () => void
  }>()
</script>
