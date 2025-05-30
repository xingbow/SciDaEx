<template>
  <div
    class="context-display"
    v-show="localQaContxtVisible"
    :style="localQaCntxtStyle"
    @mousedown="startDrag"
  >
    <el-card class="context-card" id="context-container" v-show="localQaContxtVisible">
      <div slot="header" class="header-content">
        <span style="font-size: 14px;">Relevant Contexts</span>
        <el-pagination
          layout="prev, pager, next"
          :total="contexts.length"
          :current-page.sync="localCurrentPage"
          :page-size="1"
          @current-change="handlePageChange"
        >
        </el-pagination>
        <div class="open-paper-btn">
          <el-button type="text" >Open paper</el-button>
          <el-button size="mini" style="padding: 0px !important;" @click="closeContxt"><i class="el-icon-close"></i></el-button>
        </div>
      </div>
      <div class="context-text" id="context-content">
        <div v-if="currentContext && currentContext.context && currentContext.context.type === 'text'" v-html="highlightTextSnippets(currentContext.context.content.content)">
        </div>
        <div v-else-if="currentContext && currentContext.context && currentContext.context.type === 'table'">
          <div class="demonstration">{{currentContext.context.name}}: {{ currentContext.context.caption }}</div>
          <div class="contextTable">
            <el-table size="mini" :data="contextTableData" height="200">
              <el-table-column
                fixed
                v-for="column in contextTableCols"
                :key="column.prop"
                :prop="column.prop"
                :label="column.label"
              >
                <template slot-scope="scope">
                  <span v-html="highlightTextSnippets(scope.row[column.prop])"></span>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script>
import utils from '../service/utils';

export default {
  props: {
    qaContxtVisible: {
      type: Boolean,
      required: true
    },
    qaCntxtStyle: {
      type: Object,
      required: true
    },
    contexts: {
      type: Array,
      required: true
    },
    currentPage: {
      type: Number,
      required: true
    },
    currentContext: {
      type: Object,
      required: true
    },
    contextTableData: {
      type: Array,
      required: true
    },
    contextTableCols: {
      type: Array,
      required: true
    },
    currTokens: {
      type: Array,
      default: () => []
    }
  },
  data() {
    return {
      localQaContxtVisible: this.qaContxtVisible,
      localCurrentPage: this.currentPage,
      localQaCntxtStyle: { ...this.qaCntxtStyle },
      dragging: false,
      dragStartX: 0,
      dragStartY: 0,
      initialLeft: 0,
      initialTop: 0
    };
  },
  watch: {
    qaContxtVisible(newVal) {
      this.localQaContxtVisible = newVal;
    },
    localQaContxtVisible(newVal) {
      this.$emit('update:qaContxtVisible', newVal);
    },
    currentPage(newVal) {
      this.localCurrentPage = newVal;
    },
    localCurrentPage(newVal) {
      this.$emit('update:currentPage', newVal);
    },
    qaCntxtStyle(newVal) {
      this.localQaCntxtStyle = { ...newVal };
    }
  },
  methods: {
    handlePageChange(newPage) {
      this.localCurrentPage = newPage;
    },
    closeContxt() {
      this.localQaContxtVisible = false;
    },
    highlightTextSnippets(tStr) {
      if (!utils.isEmpty(tStr)) {
        let targetStr = String(tStr);
        if (this.currTokens.length > 0 && targetStr.trim().length > 0) {
          const sortedTokens = [...this.currTokens].sort((a, b) => b.length - a.length);
          const escapedTokens = sortedTokens.map(utils.escapeRegExp);
          const regexPattern = `(${escapedTokens.join('|')})`;
          const regex = new RegExp(regexPattern, 'gi');
          targetStr = targetStr.replace(regex, match => `<span class="highlight">${match}</span>`);
          return targetStr;
        } else {
          return targetStr;
        }
      }
    },
    startDrag(event) {
      this.dragging = true;
      this.dragStartX = event.clientX;
      this.dragStartY = event.clientY;
      const rect = event.target.getBoundingClientRect();
      this.initialLeft = rect.left;
      this.initialTop = rect.top;
      document.addEventListener('mousemove', this.onDrag);
      document.addEventListener('mouseup', this.stopDrag);
    },
    stopDrag() {
      if (this.dragging) {
        this.dragging = false;
        document.removeEventListener('mousemove', this.onDrag);
        document.removeEventListener('mouseup', this.stopDrag);
        this.$emit('update:qaCntxtStyle', this.localQaCntxtStyle);
      }
    },
    onDrag(event) {
      if (this.dragging) {
        const dx = event.clientX - this.dragStartX;
        const dy = event.clientY - this.dragStartY;
        this.localQaCntxtStyle.left = `${this.initialLeft + dx}px`;
        this.localQaCntxtStyle.top = `${this.initialTop + dy}px`;
      }
    }
  }
};
</script>

<style scoped>
.context-display {
  position: absolute;
}
.highlight {
  background-color: #ff0;
}
</style>
