<template>
    
      <el-row>
        <el-col>
          <div>
            <el-tabs v-model="localActivePDFName" type="card">
              <el-tab-pane 
                v-for="(tab, index) in paperInfoList"
                :label="tab" 
                :name="tab"
                :key="index"
              ></el-tab-pane>
            </el-tabs>
            <div style="width: 100%; height: 750px; overflow: scroll;" v-show="paperInfoList.includes(localActivePDFName)">
              <div v-show="localActivePDFName=='PDF'" id="adobe-dc-view" pdfUrl="https://acrobatservices.adobe.com/view-sdk-demo/PDFs/Bodea Brochure.pdf"></div>
              <div v-show="localActivePDFName=='Table'" id="pdf-extracted-table">
                <div v-for="(table, tableIdx) in tableLists" :key="tableIdx" style="text-align: center; padding: 3px;">
                  <div class="demonstration">{{table.table_name}}: {{ table.table_caption }}</div>
                  <div :id="'tableLists-' + tableIdx" style="font-size: 13px;"></div>
                  <div class="demonstration">
                    <el-link>Mentions</el-link>
                    <el-card v-for="(tablmention, tablmentionIdx) in table.table_mentioned" :key="tablmentionIdx" style="margin: 5px;">
                      <div>
                        <span>Page-{{ tablmention.page_number }}-Sen.-{{ tablmention.sentence_number }}: </span>
                        <span>{{ tablmention.sentence_content }}</span>
                      </div>
                    </el-card>
                  </div>
                  <el-divider><i class="el-icon-star-on"></i></el-divider>
                </div>
              </div>
              <div v-show="localActivePDFName=='Figure'" id="pdf-extracted-figure">
                <div style="padding: 10px;">
                  <div class="block demo-image" v-for="(fig, figIdx) in figLists" :key="figIdx">
                    <el-image
                      style="width: 80%"
                      :src="fig.figure_url"
                      :preview-src-list="[fig.figure_url]"
                    ></el-image>
                    <div class="demonstration">{{fig.figure_name}}: {{ fig.figure_caption }}</div>
                    <div class="demonstration">
                      <el-divider><el-link>Mentions</el-link></el-divider>
                      <el-card v-for="(figmention, figmentionIdx) in fig.figure_mentioned" :key="figmentionIdx" style="margin: 5px;">
                        <div>
                          <span>Page-{{ figmention.page_number }}-Sen.-{{ figmention.sentence_number }}: </span>
                          <span>{{ figmention.sentence_content.replace(/[^\w\s\t\n\r,.!?;:()'"%-]/g, '.').replace(/\.{3,}/g, '...') }}</span>
                        </div>
                      </el-card>
                    </div>
                  </div>
                </div>
              </div>
              <div v-show="localActivePDFName=='Meta'" id="pdf-extracted-meta" style="font-size: 13px; padding: 20px;">
                <el-form size="small" label-position="left" label-width="70px">
                  <el-form-item label="Title">
                    <el-input v-model="localMetaInfo.Title" @input="updateMetaInfo('Title', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="Author">
                    <el-input v-model="localMetaInfo.Author" @input="updateMetaInfo('Author', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="Abstract">
                    <el-input v-model="localMetaInfo.Abstract" @input="updateMetaInfo('Abstract', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="Link">
                    <el-input v-model="localMetaInfo.Link" @input="updateMetaInfo('Link', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="Year">
                    <el-input v-model="localMetaInfo.Year" @input="updateMetaInfo('Year', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="Venue">
                    <el-input v-model="localMetaInfo.Journal_or_Conference" @input="updateMetaInfo('Journal_or_Conference', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="DOI">
                    <el-input v-model="localMetaInfo.DOI" @input="updateMetaInfo('DOI', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="ISSN">
                    <el-input v-model="localMetaInfo.ISSN" @input="updateMetaInfo('ISSN', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="Page">
                    <el-input v-model="localMetaInfo.Page.start" @input="updateMetaInfo('Page.start', $event)"></el-input>
                    <el-input v-model="localMetaInfo.Page.end" @input="updateMetaInfo('Page.end', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="Volume">
                    <el-input v-model="localMetaInfo.Volume" @input="updateMetaInfo('Volume', $event)"></el-input>
                  </el-form-item>
                  <el-form-item label="Publisher">
                    <el-input v-model="localMetaInfo.Publisher" @input="updateMetaInfo('Publisher', $event)"></el-input>
                  </el-form-item>
                </el-form>
              </div>
            </div>
          </div>
        </el-col>
      </el-row>
  </template>
  
  <script>
  export default {
    props: {
      activePDFName: {
        type: String,
        required: true
      },
      paperInfoList: {
        type: Array,
        required: true
      },
      tableLists: {
        type: Array,
        required: true
      },
      figLists: {
        type: Array,
        required: true
      },
      metaInfo: {
        type: Object,
        required: true
      }
    },
    data() {
      return {
        localActivePDFName: this.activePDFName,
        localMetaInfo: { ...this.metaInfo }
      };
    },
    watch: {
      activePDFName(newVal) {
        this.localActivePDFName = newVal;
      },
      localActivePDFName(newVal) {
        this.$emit('update:activePDFName', newVal);
      },
      metaInfo(newVal) {
        this.localMetaInfo = { ...newVal };
      }
    },
    methods: {
      updateMetaInfo(field, value) {
        this.localMetaInfo = { ...this.localMetaInfo, [field]: value };
        this.$emit('update:metaInfo', this.localMetaInfo);
      }
    }
  };
  </script>
  
  <style scoped>
  /* Add your component-specific styles here */
  </style>
  