<template>
    <div class="dbtableContent bg-purple-light">
      <!-- Table filtering stuff -->
      <el-row class="tablefilter-row">
        <span>Fields: </span>
        <el-select
          class="tablefilter-select"
          size="mini"
          v-model="localSelectedField"
          placeholder="Select"
        >
          <el-option
            v-for="item in fields"
            :key="item.value"
            :label="item.value"
            :value="item.value"
          ></el-option>
        </el-select>
        <span>Type: </span>
        <el-select
          class="tablefilter-select"
          size="mini"
          v-model="localSelectedOpt"
          placeholder="Select"
        >
          <el-option
            v-for="item in opts"
            :key="item.value"
            :label="item.value"
            :value="item.value"
          ></el-option>
        </el-select>
        <span>Value: </span>
        <div class="tablefilter-input">
          <el-input size="mini" placeholder="Input value" v-model="localFilterVal"></el-input>
        </div>
        <el-button class="tablefilter-button" size="mini" id="filter-clear" @click="clearFilter">Clear Filter</el-button>
        <el-button @click="downloadDB" type="info" size="mini"><i class="el-icon-download"></i></el-button>
      </el-row>
      <!-- Table content (for current DB) -->
      <div class="bg-purple-light">
        <el-row>
          <div id="db-table" class="compact" style="font-size: 13px; height: 680px;"></div>
        </el-row>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    props: {
      fields: {
        type: Array,
        required: true
      },
      opts: {
        type: Array,
        required: true
      },
      selectedField: {
        type: String,
        default: ''
      },
      selectedOpt: {
        type: String,
        default: ''
      },
      filterVal: {
        type: String,
        default: ''
      }
    },
    data() {
      return {
        localSelectedField: this.selectedField,
        localSelectedOpt: this.selectedOpt,
        localFilterVal: this.filterVal
      };
    },
    watch: {
      selectedField(newVal) {
        this.localSelectedField = newVal;
      },
      localSelectedField(newVal) {
        this.$emit('update:selectedField', newVal);
      },
      selectedOpt(newVal) {
        this.localSelectedOpt = newVal;
      },
      localSelectedOpt(newVal) {
        this.$emit('update:selectedOpt', newVal);
      },
      filterVal(newVal) {
        this.localFilterVal = newVal;
      },
      localFilterVal(newVal) {
        this.$emit('update:filterVal', newVal);
      }
    },
    methods: {
      clearFilter() {
        this.localSelectedField = '';
        this.localSelectedOpt = '';
        this.localFilterVal = '';
        this.$emit('clear-filter');
      },
      downloadDB() {
        this.$emit('download-db');
      }
    }
  };
  </script>
  
  <style scoped>
  /* Add your component-specific styles here */
  </style>
  