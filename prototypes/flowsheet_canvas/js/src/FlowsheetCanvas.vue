<script setup>
// The flowsheet canvas. Python owns the flowsheet: this component draws the
// `nodes` and `edges` it is given and reports *finished* edits as events
// (connect, move, remove, dropped, select). Dragging, panning, zooming and
// drawing a wire happen here in the browser, so they never wait on the server.
import { ref, watch, markRaw, onMounted, onBeforeUnmount } from "vue";
import { VueFlow, useVueFlow, MarkerType } from "@vue-flow/core";
import { Background } from "@vue-flow/background";
import { Controls } from "@vue-flow/controls";
import { MiniMap } from "@vue-flow/minimap";
import UnitNode from "./UnitNode.vue";
import RecycleEdge from "./RecycleEdge.vue";

const props = defineProps({
  nodes: { type: Array, default: () => [] },
  edges: { type: Array, default: () => [] },
  dragType: { type: String, default: "application/x-aquakin-unit" },
});
const emit = defineEmits(["connect", "move", "remove", "dropped", "select"]);

const flowId = "aquakin-flowsheet-" + Math.random().toString(36).slice(2);
const { screenToFlowCoordinate, fitView } = useVueFlow(flowId);
const nodeTypes = { unit: markRaw(UnitNode) };
const edgeTypes = { recycle: markRaw(RecycleEdge) };

function edgeView(e) {
  const recycle = e.data && e.data.recycle;
  return {
    ...e,
    type: recycle ? "recycle" : "smoothstep",
    animated: !!recycle,
    class: recycle ? "ak-recycle" : "",
    markerEnd: MarkerType.ArrowClosed,
    label: e.data && e.data.label,
    labelBgPadding: [6, 3],
    labelBgBorderRadius: 4,
  };
}

// Python's nodes and edges are applied to Vue Flow's store as a diff: new
// elements are added, missing ones removed, and existing ones updated in
// place. Replacing everything would rebuild every node (losing its measured
// size) and re-run edge creation against nodes not yet re-added.
const {
  addNodes, removeNodes, findNode, addEdges, removeEdges, findEdge, getNodes, getEdges,
  addSelectedNodes, addSelectedEdges, removeSelectedNodes, removeSelectedEdges, getSelectedNodes, getSelectedEdges,
} = useVueFlow(flowId);

// A plain click replaces the selection (Vue Flow otherwise keeps a node
// selected when an edge is clicked, and Delete then removes both).
function selectOnly(evt, nodes, edges) {
  if (evt && (evt.shiftKey || evt.metaKey || evt.ctrlKey)) return;
  removeSelectedNodes(getSelectedNodes.value.filter((n) => !nodes.includes(n)));
  removeSelectedEdges(getSelectedEdges.value.filter((e) => !edges.includes(e)));
}
const onEdgeClick = ({ event, edge }) => selectOnly(event, [], [edge]);
const onNodeClick = ({ event, node }) => selectOnly(event, [node], []);

function syncNodes(list) {
  const want = new Set(list.map((n) => n.id));
  const stale = getNodes.value.filter((n) => !want.has(n.id)).map((n) => n.id);
  if (stale.length) removeNodes(stale);
  const fresh = [];
  for (const n of list) {
    const cur = findNode(n.id);
    if (!cur) fresh.push({ type: "unit", ...n, data: { ...n.data } });
    else {
      cur.data = { ...n.data };
      if (!cur.dragging) cur.position = { ...n.position };
    }
  }
  if (fresh.length) addNodes(fresh);
}

function syncEdges(list) {
  const want = new Set(list.map((e) => e.id));
  const stale = getEdges.value.filter((e) => !want.has(e.id)).map((e) => e.id);
  if (stale.length) removeEdges(stale);
  const fresh = [];
  for (const e of list) {
    const view = edgeView(e);
    const cur = findEdge(e.id);
    if (!cur) fresh.push(view);
    else Object.assign(cur, { type: view.type, data: view.data, label: view.label, animated: view.animated, class: view.class });
  }
  if (fresh.length) addEdges(fresh);
}

// Nodes first: an edge to a new node needs the node in the store.
watch(() => props.nodes, (v) => syncNodes(v || []), { immediate: true, deep: true });
watch(() => props.edges, (v) => syncEdges(v || []), { immediate: true, deep: true });

// Keep the whole flowsheet in view until the user pans or zooms: refit once
// the nodes are measured and whenever the canvas is resized (the page lays
// out after the first fit, so a single fit frames the wrong width).
const container = ref(null);
let autoFit = true;
let measured = false;
const refit = () => autoFit && measured && fitView({ padding: 0.06 });
function onNodesInitialized() {
  measured = true;
  refit();
}
function onMoveStart(e) {
  if (e && e.event) autoFit = false;
}
let observer;
onMounted(() => {
  observer = new ResizeObserver(() => refit());
  observer.observe(container.value);
});
onBeforeUnmount(() => observer && observer.disconnect());

function onConnect(c) {
  emit("connect", {
    source: c.source,
    sourceHandle: c.sourceHandle,
    target: c.target,
    targetHandle: c.targetHandle,
  });
}

function onDragStop(evt) {
  emit("move", evt.nodes.map((n) => ({ id: n.id, position: { x: Math.round(n.position.x), y: Math.round(n.position.y) } })));
}

// Vue Flow removes deleted elements itself; report them so Python can follow.
// Only elements Python still has are reported, so the removals made while
// applying Python's own update (a rename, say) are not echoed back.
const known = (list, id) => (list || []).some((x) => x.id === id);
function onNodesChange(changes) {
  const ids = changes.filter((c) => c.type === "remove" && known(props.nodes, c.id)).map((c) => c.id);
  if (ids.length) emit("remove", { nodes: ids, edges: [] });
}
function onEdgesChange(changes) {
  const ids = changes.filter((c) => c.type === "remove" && known(props.edges, c.id)).map((c) => c.id);
  if (ids.length) emit("remove", { nodes: [], edges: ids });
}

function onSelection({ nodes, edges }) {
  emit("select", { nodes: (nodes || []).map((n) => n.id), edges: (edges || []).map((e) => e.id) });
}

function onDragOver(evt) {
  if (evt.dataTransfer.types.includes(props.dragType)) {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = "copy";
  }
}
function onDrop(evt) {
  const kind = evt.dataTransfer.getData(props.dragType);
  if (!kind) return;
  evt.preventDefault();
  const p = screenToFlowCoordinate({ x: evt.clientX, y: evt.clientY });
  emit("dropped", { kind, position: { x: Math.round(p.x), y: Math.round(p.y) } });
}
</script>

<template>
  <div ref="container" class="ak-canvas" @dragover="onDragOver" @drop="onDrop">
    <VueFlow
      :id="flowId"
      :node-types="nodeTypes"
      :edge-types="edgeTypes"
      :delete-key-code="['Delete', 'Backspace']"
      :snap-to-grid="true"
      :snap-grid="[10, 10]"
      :min-zoom="0.2"
      :max-zoom="2.5"
      @nodes-initialized="onNodesInitialized"
      @move-start="onMoveStart"
      @connect="onConnect"
      @edge-click="onEdgeClick"
      @node-click="onNodeClick"
      @node-drag-start="onNodeClick"
      @node-drag-stop="onDragStop"
      @nodes-change="onNodesChange"
      @edges-change="onEdgesChange"
      @selection-change="onSelection"
    >
      <Background :gap="20" pattern-color="var(--ak-grid)" />
      <Controls position="bottom-left" />
      <MiniMap position="bottom-right" :width="160" :height="100" pannable zoomable :node-color="(n) => (n.data && n.data.colour) || '#888'" />
    </VueFlow>
  </div>
</template>
