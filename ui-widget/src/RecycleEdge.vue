<script setup>
// A recycle (back) edge: leaves its source to the right, drops to its own lane
// below the units it spans, runs back left, and rises into its target. The
// lane is a flow coordinate Python assigns per loop, so loops never run along
// the forward line or on top of each other.
import { computed } from "vue";
import { BaseEdge } from "@vue-flow/core";

const props = defineProps({
  id: String, sourceX: Number, sourceY: Number, targetX: Number, targetY: Number,
  data: Object, markerEnd: String, style: Object, label: [String, Object],
  labelStyle: Object, labelShowBg: Boolean, labelBgStyle: Object,
  labelBgPadding: Array, labelBgBorderRadius: Number,
});

const geometry = computed(() => {
  const { sourceX: sx, sourceY: sy, targetX: tx, targetY: ty } = props;
  const lane = props.data && props.data.lane != null ? props.data.lane : Math.max(sy, ty) + 80;
  const r = 8;
  const ox = sx + 22;
  const ix = tx - 22;
  const path = [
    `M ${sx} ${sy}`, `L ${ox - r} ${sy}`, `Q ${ox} ${sy} ${ox} ${sy + r}`,
    `L ${ox} ${lane - r}`, `Q ${ox} ${lane} ${ox - r} ${lane}`,
    `L ${ix + r} ${lane}`, `Q ${ix} ${lane} ${ix} ${lane - r}`,
    `L ${ix} ${ty + r}`, `Q ${ix} ${ty} ${ix + r} ${ty}`, `L ${tx} ${ty}`,
  ].join(" ");
  return { path, labelX: (ox + ix) / 2, labelY: lane };
});
</script>

<template>
  <BaseEdge
    :id="id" :path="geometry.path" :marker-end="markerEnd" :style="style"
    :label="label" :label-x="geometry.labelX" :label-y="geometry.labelY"
    :label-bg-padding="labelBgPadding" :label-bg-border-radius="labelBgBorderRadius"
  />
</template>
