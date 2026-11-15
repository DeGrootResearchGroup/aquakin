<script setup>
// One plant unit: a header naming it and one row per port. Every port is a
// Vue Flow handle whose id is the port name, so a connection reads directly as
// aquakin's "unit.port" -> "unit.port".
import { inject, computed } from "vue";
import { Handle, Position, useVueFlow } from "@vue-flow/core";

const props = defineProps({ id: String, data: Object, selected: Boolean });
const { getEdges } = useVueFlow();
const icons = inject("akIcons", computed(() => ({})));
const iconSvg = computed(() => icons.value[props.data.icon] || "");

// Instant feedback while a wire is drawn (an input is fed once, an output
// feeds one unit). It sits on the handles so it judges only wires being drawn,
// never edges loaded from Python, which stays the authority on every edit.
function isValidConnection(c) {
  if (c.source === c.target) return false;
  return !getEdges.value.some(
    (e) =>
      (e.target === c.target && e.targetHandle === c.targetHandle) ||
      (e.source === c.source && e.sourceHandle === c.sourceHandle)
  );
}

const isOpen = (port) => (props.data.open || []).includes(port);
const isDangling = (port) => (props.data.dangling || []).includes(port);
</script>

<template>
  <div class="ak-unit" :class="{ 'ak-selected': selected, 'ak-has-error': !!data.error }" :style="{ '--ak-accent': data.colour || '#5b8def' }">
    <div class="ak-head">
      <span v-if="iconSvg" class="ak-icon" v-html="iconSvg" />
      <div class="ak-titles">
        <span class="ak-name">{{ data.label || id }}</span>
        <span class="ak-kind">{{ data.kindLabel }}</span>
      </div>
      <span v-if="data.error" class="ak-error" :title="data.error">!</span>
    </div>
    <div v-if="data.summary" class="ak-summary">{{ data.summary }}</div>
    <div class="ak-ports">
      <div v-for="p in data.inputs" :key="'in-' + p" class="ak-port ak-in" :class="{ 'ak-open': isOpen(p) }">
        <Handle :id="p" type="target" :position="Position.Left" class="ak-handle" :is-valid-connection="isValidConnection" />
        <span>{{ p }}</span>
      </div>
      <div v-for="p in data.outputs" :key="'out-' + p" class="ak-port ak-out" :class="{ 'ak-open': isOpen(p), 'ak-dangling': isDangling(p) }">
        <span>{{ p }}</span>
        <Handle :id="p" type="source" :position="Position.Right" class="ak-handle" :is-valid-connection="isValidConnection" />
      </div>
    </div>
  </div>
</template>
