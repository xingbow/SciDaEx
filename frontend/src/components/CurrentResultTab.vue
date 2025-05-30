<template>
  <div class="bg-purple-light">
    <!-- Table filtering stuff -->
    <el-row class="tablefilter-row">
      <span>Fields: </span>
      <el-select
        class="tablefilter-select"
        size="mini"
        v-model="localQaSelectedField"
        placeholder="Select"
      >
        <el-option
          v-for="(item, index) in qaFields"
          :key="index"
          :label="item.value"
          :value="item.value"
        ></el-option>
      </el-select>
      <span>Type: </span>
      <el-select
        class="tablefilter-select"
        size="mini"
        v-model="localQaSelectedOpt"
        placeholder="Select"
      >
        <el-option
          v-for="(item, index) in opts"
          :key="index"
          :label="item.value"
          :value="item.value"
        ></el-option>
      </el-select>
      <span>Value: </span>
      <div class="tablefilter-input">
        <el-input size="mini" placeholder="Input value" v-model="localQaFilterVal"></el-input>
      </div>
      <el-button class="tablefilter-button" size="mini" id="qatable-filter-clear" @click="QAclearFilter">Clear Filter</el-button>
      <el-button type="primary" size="mini" icon="el-icon-arrow-right" @click="addToDB">Add to DB</el-button>
      
    </el-row>
    <el-row class="tablefilter-row">
      <!-- <span class='fa-solid fa-minus'></span>:{{recordChecked.normelCount}} -->
      <i class='fa-solid fa-circle-exclamation' style='color: crimson;'></i>:{{recordChecked.alertCount}}
      <!-- <span class='fa-solid fa-check' style='color: green;'></span>:{{recordChecked.checkedCount}} -->
    </el-row>
    <el-row>
      <div id="qa-table" class="compact" style="font-size: 13px; height: 680px;"></div>
    </el-row>
  </div>
</template>

<script>
export default {
  props: {
    qaFields: {
      type: Array,
      required: true
    },
    opts: {
      type: Array,
      required: true
    },
    qaSelectedField: {
      type: String,
      default: ''
    },
    qaSelectedOpt: {
      type: String,
      default: ''
    },
    qaFilterVal: {
      type: String,
      default: ''
    },
    recordChecked: {
      type: Object,
      default: () => ({})
    }
  },
  data() {
    return {
      localQaSelectedField: this.qaSelectedField,
      localQaSelectedOpt: this.qaSelectedOpt,
      localQaFilterVal: this.qaFilterVal,
      localrecordChecked: this.recordChecked,
    };
  },
  watch: {
    qaSelectedField(newVal) {
      this.localQaSelectedField = newVal;
    },
    localQaSelectedField(newVal) {
      this.$emit('update:qaSelectedField', newVal);
    },
    qaSelectedOpt(newVal) {
      this.localQaSelectedOpt = newVal;
    },
    localQaSelectedOpt(newVal) {
      this.$emit('update:qaSelectedOpt', newVal);
    },
    qaFilterVal(newVal) {
      this.localQaFilterVal = newVal;
    },
    localQaFilterVal(newVal) {
      this.$emit('update:qaFilterVal', newVal);
    },
  },
  methods: {
    QAclearFilter() {
      this.localQaSelectedField = '';
      this.localQaSelectedOpt = '';
      this.localQaFilterVal = '';
      this.$emit('qa-clear-filter');
    },
    addToDB() {
      this.$emit('add-to-db');
    },
    changeAlert(){
      this.localrecordChecked = {
        normelCount: 0,
        alertCount: 1,
        checkedCount: 2
      }
      this.$emit('update:recordChecked', this.localrecordChecked);
    }
  }
};
</script>

<style scoped>
/* Add your component-specific styles here */
</style>
