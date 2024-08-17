<template>
  <div id="app" style="width: 100%;">
    <el-row type="flex">
      <el-col :span="sidebarSpan" style="position: relative; ">
        <el-header style="height: 50px !important;">
          <div class="header-web">
            SciDaEx
          </div>
        </el-header>

        <!-- PDF upload components -->
        <el-row>
          <el-upload ref="upload" :action="uploadUrl" :auto-upload="false" multiple :on-change="handleFileChange"
            :before-upload="beforeUpload" accept=".pdf" list-type="text" :show-file-list="false">
            <el-button size="mini" type="info">Upload PDF files</el-button>
            <el-button size="mini" plain disabled>Import from directory</el-button>
            <div slot="tip" class="el-upload__tip">PDF files with a size less than 250mb</div>
          </el-upload>
        </el-row>

        <el-tabs v-model="activeName" style="padding: 0px !important;">
          <el-tab-pane label="Paper lists" name="1" style="padding: 0px !important;">
            <div style="padding:2px;">
              <div id="pdf-table" class="compact" style="font-size: 13px; height: 640px; margin: 0px !important;"></div>
            </div>
          </el-tab-pane>
          <el-tab-pane label="Query" name="2">
            <el-row v-loading="loadingAIMessage"
              style="text-align: left; display: flex; flex-direction: column; height: 660px; position:relative;">
              <!-- Messages Container -->
              <div class="messages"
                style="flex-grow: 1; overflow-y: auto; padding: 5px; height: 650px; margin-bottom:10px;">
                <!-- Individual Message Containers -->
                <div v-for="(message, index) in messages" :key="index" class="message-row">
                  <!-- Display user icon for user messages -->
                  <i v-if="isUserMessage(index)" class="el-icon-user user-icon"></i>
                  <!-- Display AI icon for AI messages -->
                  <i v-else class="el-icon-chat-dot-square ai-icon"></i>
                  <!-- Message Container -->
                  <div class="message-container"
                    :class="{ 'user-message': isUserMessage(index), 'ai-message': !isUserMessage(index) }">
                    {{ message }}
                  </div>
                </div>
              </div>
              <add-additional-columns @updateUserColumnInput="updateUserColumnInput"></add-additional-columns>
              <div class="chat-container" style="position: absolute; bottom: 0; width: 20vw; margin-bottom: 10px;">
                <textarea style="width: 90%" v-model="userinput" @keyup.enter="sendUserMessage"
                  placeholder="Type a message..."></textarea>
                <el-button size="mini" style="position: absolute; right: 0; top: 0; height: 100%;"
                  @click="sendUserMessage" plain><i class="el-icon-position"
                    style="transform: rotate(45deg)"></i></el-button>
              </div>

            </el-row>
          </el-tab-pane>
        </el-tabs>

      </el-col>

      <el-col :span="contentSpan" class="content-col">

        <!-- Button at the left edge of the right column -->
        <div class="toggle-button-container">
          <el-tooltip class="item" effect="dark" :content="isSidebarOpen ? 'Hide Sidebar' : 'Display Sidebar'"
            placement="right" v-model="isTooltipVisible">
            <button @click="toggleSidebar" class="sidebar-toggle-button">
              <i class="el-icon-arrow-left" v-if="isSidebarOpen"></i>
              <i class="el-icon-arrow-right" v-else></i>
            </button>
          </el-tooltip>
        </div>
        <!-- Tables of current results and database are here -->
        <el-row id="main-view">
          <el-col :span="12">
            <el-row>
              <el-col>
                <div style="padding-right: 40px;">
                  <el-tabs v-model="activeTabName" type="card">
                    <el-tab-pane label="Current Result" name="qa" style="padding: 0px !important;">
                      <!-- table content (for current question)-->
                      <current-result-tab :qaFields="qaFields" :opts="opts" :qaSelectedField.sync="qaSelectedField"
                        :qaSelectedOpt.sync="qaSelectedOpt" :qaFilterVal.sync="qaFilterVal"
                        :recordChecked.sync="recordChecked" @qa-clear-filter="QAclearFilter" @add-to-db="addToDB" />
                    </el-tab-pane>
                    <el-tab-pane label="DB" name="db" style="padding: 0px !important;">
                      <db-tab :fields="fields" :opts="opts" :selectedField.sync="selectedField"
                        :selectedOpt.sync="selectedOpt" :filterVal.sync="filterVal" @clear-filter="clearFilter"
                        @download-db="downloadDB" />
                    </el-tab-pane>
                  </el-tabs>
                </div>
              </el-col>
            </el-row>
          </el-col>
          <!-- PDF viewer -->
          <el-col :span="12">
            <paper-viewer :activePDFName.sync="activePDFName" :paperInfoList="paperInfoList" :tableLists="tableLists"
              :figLists="figLists" :metaInfo="metaInfo" :selectedFile="selectedFile" />
          </el-col>

        </el-row>

      </el-col>
    </el-row>

    <!-- handle summarization & context menu -->
    <el-dialog :visible.sync="dialogVisible" title="Summary">
      <div v-loading="summaryLoading" element-loading-text="Loading summary..."
        element-loading-spinner="el-icon-loading" element-loading-background="rgba(0, 0, 0, 0.8)">{{ summary }}</div>
    </el-dialog>

    <div style="display:none" class="context-menu">
      <ul>
        <li @click="handleMenuOption('summarize')">Summarize</li>
      </ul>
    </div>

    <div id="confidencePopup" v-show="showAlert" :style="showAlertStyle">
      <strong style='font-size:1.2em;'>Low confidence</strong>
      <div>LLM may be incorrect here. Please double check!</div>
    </div>

    <!-- system login -->
    <SystemLogin :dialog-form-visible.sync="dialogFormVisible" :form="loginForm" :form-label-width="formLabelWidth"
      :passwd="loginPasswd"></SystemLogin>

    <!-- context viewer -->
    <context-viewer :qaContxtVisible.sync="qaContxtVisible" :qaCntxtStyle="qaCntxtStyle" :contexts="contexts"
      :currentPage.sync="currentPage" :currentContext="currentContext" :contextTableData="contextTableData"
      :contextTableCols="contextTableCols" :currTokens="currTokens" @update:qaCntxtStyle="updateQaCntxtStyle" />

  </div>
</template>

<script>

/* global d3 $ _ */  // eslint-disable-line
import { TabulatorFull as Tabulator } from 'tabulator-tables'; //import Tabulator library
import service from './service/service.js';
import utils from "./service/utils.js"
// import components
import SystemLogin from './components/SystemLogin';
import CurrentResultTab from './components/CurrentResultTab';
import DbTab from './components/DbTab';
import PaperViewer from './components/PaperViewer';
import ContextViewer from './components/ContextViewer';
import AddAdditionalColumns from './components/AddAdditionalColumns';

// NLP processing toolkit (wink-tokenizer & stopword removal)
const { removeStopwords } = require('stopword')
var tokenizer = require('wink-tokenizer');
var myTokenizer = tokenizer();

export default {
  name: 'App',
  components: {
    SystemLogin,
    CurrentResultTab,
    DbTab,
    PaperViewer,
    ContextViewer,
    AddAdditionalColumns
  },
  data() {
    return {
      isTableExpanded: false,
      isTableToolTipVsisible: false,
      isSidebarOpen: true,
      isTooltipVisible: false, // Controls the visibility of the tooltip
      activeModule: null,
      userinput: "",
      adobeDCView: null,
      messages: [],
      loadingAIMessage: false,
      drawer: false,
      // pdfTable: null,

      uploadUrl: service.dataServerUrl + '/upload',
      // pdf processing
      selectedFile: null,
      fileTabletoRerun: [],
      fileList: [],
      activeName: "1",
      pdfUrlPrefix: service.dataServerUrl + '/uploads',
      numRows: 3,

      // db table settings
      selectedField: "",
      fields: [],
      selectedOpt: "",
      opts: [
        { "value": "=" },
        { "value": "<" },
        { "value": "<=" },
        { "value": ">" },
        { "value": ">=" },
        { "value": "!=" },
        { "value": "contain" },
      ],
      filterVal: "",
      // qa table setting
      qaSelectedField: "",
      qaFields: [],
      qaSelectedOpt: "",
      qaFilterVal: "",
      // openai key
      openai_key: "",
      // pdf info
      tableLists: [],
      metaInfo: {
        "Title": "",
        "Author": "",
        "Abstract": "",
        "Link": "",
        "Year": "",
        "Journal_or_Conference": "",
        "DOI": "",
        "ISSN": "",
        "Page": {
          "start": "",
          "end": ""
        },
        "Volume": "",
        "Publisher": ""
      },
      figLists: [],
      // pdf categories
      pdfCategories: null,
      // context menu and summarization
      showAlert: false,
      showAlertStyle: {},
      dialogVisible: false,
      tobeSummarized: [],
      summary: "",
      summaryLoading: true,

      // qa context display
      currTokens: [],
      qaContexts: {},
      qaContxtVisible: false,
      qaCntxtStyle: {},
      currentCntxtPage: 1,
      contexts: [],
      currentPage: 1,
      pageSize: 5,
      contextTableData: [],
      contextTableCols: [],

      // tabs on the right
      activeTabName: 'db',
      activePDFName: "PDF",
      editableTabs: ["PDF",],
      paperInfoList: ['PDF',],

      // active current table or current db
      activeDBName: "db",
      qa_tableData: [],

      dbTableExpanded: false,
      qaTableExpanded: false,

      scatterPlotLoading: false,

      // login form
      dialogFormVisible: false,
      loginForm: {
        name: '',
        userid: '',
      },
      formLabelWidth: '120px',
      loginPasswd: "",
      userInteractions: {
        "systemname": "SciDaSynth",
        "projection_interactions": [],
        "table_interactions": [],
      },

      // alert count (alertCount, normelCount, checkedCount)
      recordChecked: {
        alertCount: 0,
        normelCount: 0,
        checkedCount: 0,
      },
    }
  },
  computed: {
    currentContext() {
      return {
        "context": this.contexts[this.currentPage - 1],
        "visible": this.qaContxtVisible,
        "visibleStyle": this.qaCntxtStyle
      }
    },
    tableFilter: function () {
      return {
        "field": this.selectedField,
        "type": this.selectedOpt,
        "value": this.filterVal
      }
    },
    qaTableFilter: function () {
      return {
        "field": this.qaSelectedField,
        "type": this.qaSelectedOpt,
        "value": this.qaFilterVal
      }
    },
    // left panel shown or hidden
    sidebarSpan() {
      return this.isSidebarOpen ? 5 : 0;
    },
    contentSpan() {
      return this.isSidebarOpen ? 19 : 24;
    },
  },
  watch: {
    currentContext(currentContext) {
      let contextContent = currentContext.context;
      console.log("contextContent: ", contextContent);
      if (contextContent.type == "table") {
        let tableContent = JSON.parse(contextContent.content);
        console.log("tableContent", tableContent);
        this.contextTableData = tableContent;

        if (tableContent.length > 0) {
          this.contextTableCols = Object.keys(tableContent[0]).map(key => ({
            label: key,
            prop: key
          }));
        } else {
          this.contextTableCols = [];
        }
      }

    },
    activeTabName(activeTabName) {
      // console.log("activeTabName", activeTabName);
      if (activeTabName == "db" || activeTabName == "qa") {
        this.handleModuleSelect("0")
      }
    },
    dialogVisible(dialogVisible) {
      if (dialogVisible == false) {
        this.summaryLoading = true;
      }
    },
    fileList(fileList) {
      const _this = this;
      console.log("fileList changed", fileList);
      let filenames = fileList.map((file) => {
        return file.name;
      });
      console.log("filenames for clustering pdfs", filenames);

      let dbData = this.dbTable.getData();
      service.extract_meta_from_pdf(fileList, this.openai_key, (metaInfos) => {
        // meta information
        let pdf_meta = [];
        metaInfos.map((meta, i) => {
          let pdf_file = fileList[i].name;
          pdf_meta.push({
            "pdf_file": pdf_file,
            "citation": utils.formatCitationAPA(meta),
            "url": meta.Link
          })
        });
        // join with current db table
        // console.log("meta data & db data:", pdf_meta, dbData);
        let merged_dbData = utils.joinArrays(dbData, pdf_meta, ["pdf_file"], "outer")
        console.log("merged_dbData", merged_dbData);

        _this.dbTable = new Tabulator("#db-table", {
          data: merged_dbData,
          autoColumns: true,
          columnDefaults: {
            maxWidth: 250, //maximum column width of 300px for all columns
          },

          autoColumnsDefinitions: function (definitions) {
            definitions.forEach((column) => {
              column.editor = true;
              column.cellClick = function (e, cell) {
                let filename = cell.getRow().getData().pdf_file;
                let idx = _this.fileList.findIndex((file) => {
                  return file.name == filename;
                })
                _this.selectedFile = _this.fileList[idx].raw;
              }
            });

            return definitions;
          },
        });
        _this.fields = Object.keys(merged_dbData[0]).map(field => {
          return { "value": field }
        })

      })
    },
    selectedFile(selectedFile) {
      console.log("selectedFile changed", this.pdfUrlPrefix + '/' + this.selectedFile.name);
      if (this.selectedFile != null) {
        // this.renderPDFbySelectedFile(selectedFile);
        // display PDF info.
        this.displayPDFTable(selectedFile.name);
        this.displayPDFFigure(selectedFile.name);
        this.displayPDFMeta(selectedFile.name);
      }
    },
    tableFilter: function () {
      if (this.selectedField != "" && this.selectedOpt != "" && this.filterVal != "") {
        console.log("this.tableFilter changed", this.tableFilter);
        this.dbTable.setFilter(this.selectedField, this.selectedOpt, this.filterVal);
      }
    },
    qaTableFilter: function () {
      if (this.qaSelectedField != "" && this.qaSelectedOpt != "" && this.qaFilterVal != "") {
        console.log("this.qaTableFilter changed", this.qaTableFilter);
        this.qaTable.setFilter(this.qaSelectedField, this.qaSelectedOpt, this.qaFilterVal);
        // record user interactions
        this.userInteractions.table_interactions.push({
          "time": new Date().getTime(),
          "type": "filter",
          "view": "current_results",
          "target": "tablecell"
        })
      }
    },
    tableLists(tableLists) {
      // console.log("do nothing")
      this.$nextTick(() => {
        tableLists.map((t, i) => {
          new Tabulator("#tableLists-" + i, {
            data: JSON.parse(t.table_content),
            autoColumns: true,
            autoColumnsDefinitions: function (definitions) {
              definitions.forEach((column) => {
                column.editor = true;
                column.editableTitle = true;
              });

              return definitions;
            }
          });
        })
      })

    },
  },
  methods: {
    beforeUpload() {
      return false; // Prevent auto-upload
    },
    loadFiles() {
      service.getFiles((files) => {
        this.fileList = files.map(file => {
          let rawFile = null;
          if (file.raw) {
            try {
              rawFile = this.base64ToFile(file.raw, file.name);
            } catch (error) {
              console.error(`Failed to convert base64 to file for ${file.name}:`, error);
            }
          }
          return {
            name: file.name,
            url: file.url,
            raw: rawFile
          };
        });

        // Populate the PDF table with the files
        this.fileList.forEach(file => {
          this.pdfTable.addRow({
            name: file.name,
            url: file.url
          }, false);
        });

        // If you want to automatically select the first file
        if (this.fileList.length > 0) {
          this.selectedFile = this.fileList[0].raw;
        }
      });
    },

    base64ToFile(base64String, filename) {
      if (!base64String) {
        console.error(`No base64 string provided for file ${filename}`);
        return null;
      }

      try {
        const arr = base64String.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
          u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], filename, { type: mime });
      } catch (error) {
        console.error(`Failed to convert base64 to file for ${filename}:`, error);
        return null;
      }
    },
    async uploadFiles() {
      const promises = this.fileList.map(file => this.$refs.upload.submit(file));
      await Promise.all(promises);
      this.userInteractions.startTime = new Date().getTime();
      console.log('All files uploaded at:', new Date().getTime());
    },
    alertIconFormatter(cell) {
      // console.log("alertIconFormatter", cell);
      let currRowData = cell.getRow().getData();
      let total_records = 0, alert_records = 0;
      Object.keys(currRowData).map((key) => {
        if (key != "pdf_file") {
          total_records += 1;
          if (utils.isEmpty(currRowData[key]) || String(currRowData[key]).trim().toLowerCase() == "empty") {
            alert_records += 1;
          }
        }
      });
      if (alert_records / total_records >= service.alertTh) {
        return "<span class='fa-solid fa-circle-exclamation' style='color: red;'></span>";
      } else {
        return "";
      }
    },
    handleModuleSelect(index) {
      console.log("handleModuleSelect", this.activeModule, index);
      this.activeModule = index;

      const tabNames = ["PDF", "Table", "Figure", "Meta"];
      const tabName = tabNames[index - 1];
      console.log("===tagName in handleModuleSelect===", tabName);

      if (tabName && !this.editableTabs.includes(tabName)) {
        this.editableTabs.push(tabName);
      }

      if (tabName) {
        console.log("tagName in handleModuleSelect", tabName);
        this.activeTabName = tabName;
      }

      if (tabNames.includes(this.activeTabName)) {
        // Record user interactions
        this.userInteractions.table_interactions.push({
          time: new Date().getTime(),
          type: "open_context",
          view: "current_results",
          target: this.activeTabName
        });
      }
    },
    isUserMessage(index) {
      // Assuming user messages are at odd indexes
      return index % 2 == 0;
    },
    sendUserMessage() {
      console.log('Sending message:', this.userinput);
      const _this = this;
      if (this.userinput.trim() !== "") {
        if (this.fileList.length > 0) {
          _this.loadingAIMessage = true;
          // fake response here
          let userinput = JSON.parse(JSON.stringify(this.userinput));
          this.messages.push(this.userinput);
          let fileSamples = this.fileList;
          // _.sampleSize(this.fileList, 5); // TODO: change the sample size here
          console.log("fileSamples", fileSamples.length, fileSamples);
          // DONE: handle user question at the backend
          service.run_qa(this.userinput, fileSamples, (response) => {
            _this.activeDBName = "table";
            _this.loadingAIMessage = false;
            console.log("received response (for user question)", response);
            this.messages.push(response.summary);

            let qa_tableData = [];
            let qa_cntxts = {};
            // reset recordChecked
            _this.recordChecked = {
              alertCount: 0,
              normelCount: 0,
              checkedCount: 0,
            }
            // add qa table given the response
            for (let key in response.answer) {
              let pdf_file = key;
              let answers = response.answer[pdf_file].answer;
              // console.log("response.answer[pdf_file]: ", response.answer[pdf_file])
              let cntxt = response.answer[pdf_file].context;
              let cntxt_lists = []
              if (cntxt.text.length > 0) {
                cntxt.text.map(t => {
                  cntxt_lists.push({
                    "type": "text",
                    "content": t
                  })
                })
              }
              if (cntxt.tables.length > 0) {
                cntxt.tables.map(t => {
                  cntxt_lists.push(_.extend(t, {
                    "type": "table"
                  }))
                })
              }
              if (cntxt.figures.length > 0) {
                cntxt.figures.map(t => {
                  cntxt_lists.push(_.extend(t, {
                    "type": "figure"
                  }))
                })
              }

              qa_cntxts[pdf_file] = cntxt_lists

              answers.map(answer => {
                // // add column: "checked" 
                // answer["checked"] = utils.isAlert(answer);
                if (utils.isAlert(answer)) {
                  _this.recordChecked.alertCount += 1;
                } else {
                  _this.recordChecked.normelCount += 1;
                }
                qa_tableData.push(_.extend(answer, {
                  "pdf_file": pdf_file,
                }))
              })

            }

            // add QA contexts given the response
            _this.qaContexts[userinput] = qa_cntxts

            console.log("qa_tableData", qa_tableData);
            _this.activeDBName = "table";
            _this.qa_tableData = qa_tableData;
            // set up qa table
            if (_this.qaTable) {
              _this.qaTable.destroy();
            }
            _this.qaTable = new Tabulator("#qa-table", {
              data: qa_tableData,
              autoColumns: true,
              rowContextMenu: [
                {
                  label: '<el-button type="primary">Open context</el-button>',
                  action: function (e, row) {
                    _this.handleOpenContext(e, row);
                    //  record user interactions
                    _this.userInteractions.table_interactions.push({
                      "time": new Date().getTime(),
                      "type": "open_context",
                      "view": "current_results",
                      "target": "tablecell"
                    })
                  }
                },
                {
                  label: "Open paper",
                  action: function (_e, row) {
                    let cell = row.getCells()[0];
                    _this.handleTableRightClickMenuOptions(cell);
                    _this.activePDFName = "PDF"
                  }
                },
                {
                  label: "Open table",
                  action: function (_e, row) {
                    let cell = row.getCells()[0];
                    _this.handleTableRightClickMenuOptions(cell);
                    if (!_this.paperInfoList.includes("Table")) {
                      _this.paperInfoList.push("Table")
                    }
                    _this.activePDFName = "Table"
                  }
                },
                {
                  label: "Open figure",
                  action: function (_e, row) {
                    let cell = row.getCells()[0];
                    _this.handleTableRightClickMenuOptions(cell);
                    if (!_this.paperInfoList.includes("Figure")) {
                      _this.paperInfoList.push("Figure")
                    }
                    _this.activePDFName = "Figure"
                  }
                },
                {
                  label: "Open meta",
                  action: function (_e, row) {
                    let cell = row.getCells()[0];
                    _this.handleTableRightClickMenuOptions(cell);
                    if (!_this.paperInfoList.includes("Meta")) {
                      _this.paperInfoList.push("Meta")
                    }
                    _this.activePDFName = "Meta"
                  }
                }
              ],
              columnDefaults: {
                maxWidth: 250, //maximum column width of 300px for all columns
              },
              autoColumnsDefinitions: function (definitions) {
                definitions.unshift(
                  { formatter: "rowSelection", titleFormatter: "rowSelection", hozAlign: "center", headerSort: false, frozen: true, width: 30 },
                  {
                    title: "",
                    formatter: _this.alertIconFormatter,
                    maxWidth: 40,
                    hozAlign: "center",
                    cellMouseLeave: function () {
                      _this.showAlert = false
                    },
                    // cellMouseOver: function (e) {
                    //   if (!utils.isEmpty(e.target.innerHTML)) {
                    //     console.log("e, component, onRendered: ", e.target.innerHTML)
                    //     _this.showAlert = true
                    //     let left = e.clientX, top = e.clientY
                    //     _this.showAlertStyle = {
                    //       position: 'absolute',
                    //       top: `${top}px`,
                    //       left: `${left}px`,
                    //       'text-align': 'left',
                    //       'font-size': '12px',
                    //       'background-color': 'white',
                    //       "padding": "10px",
                    //       "border": "1px solid #ccc",
                    //     };
                    //   } else {
                    //     _this.showAlert = false
                    //   }
                    // },
                    clickMenu: function (e) {
                      console.log("cellClick", e, e.target.innerHTML, utils.isEmpty(e.target.innerHTML));
                      _this.showAlert = false
                      if (!utils.isEmpty(e.target.innerHTML)) {
                        return [
                          {
                            label: "Checked",
                            action: function () {
                              e.target.innerHTML = "";
                              _this.recordChecked.checkedCount += 1;
                              _this.recordChecked.alertCount -= 1;
                            }
                          },
                        ]
                      }
                    }
                  },
                )
                definitions.forEach((column) => {
                  if (column.field != "pdf_file" && (!utils.isEmpty(column.field))) { // get columns except pdf_file and empty columns
                    // column.editor = true;
                    column.editor = "input";
                    column.editable = false;

                    column.cellDblClick = function (e, cell) {
                      // _this.handleTableRightClickMenuOptions(cell);
                      cell.edit(true);
                    }

                    column.editableTitle = true;
                    column.formatter = "textarea";
                    column.headerMenu = function () {
                      var menu = [
                        {
                          label: 'Set value for active rows',
                          action: () => {
                            const newValue = prompt(`Enter new value for ${column.field}:`);
                            if (newValue !== null) {
                              ///////////////////////////////////////////////
                              // record user interactions
                              _this.userInteractions.table_interactions.push({
                                "time": new Date().getTime(),
                                "type": "batch_edit",
                                "view": "current_results",
                                "target": "tablecell"
                              })
                              ///////////////////////////////////////////////
                              if (this.getSelectedRows().length > 0) {
                                this.getSelectedRows().forEach(row => {
                                  row.update({ [column.field]: newValue });
                                  row.deselect();
                                });
                              } else {
                                this.getRows("active").forEach(row => {
                                  row.update({ [column.field]: newValue });
                                });
                              }

                              // this.updateActiveRows(column.field, newValue);
                            }
                          }
                        },
                        {
                          label: 'Add a new column',
                          action: () => {
                            const newColumn = prompt(`Enter new column name:`);
                            if (newColumn !== null) {
                              this.addColumn({
                                title: newColumn,
                                field: newColumn,
                                editor: true,
                                editableTitle: true,
                                formatter: "textarea"
                              }, true, column.field);
                            }
                          }
                        }
                      ];

                      return menu;
                    };
                    _this.qaFields.push({
                      "value": column.title
                    })
                    // handle cell color, highlight empty cell
                    column.formatterParams = function (cell) {
                      utils.highlightcells(cell)

                    }
                    column.cellEdited = function (cell) {
                      utils.highlightcells(cell)
                    }
                  }

                });
                // 
                _this.qaFields = _.uniq(_this.qaFields);
                return definitions;
              },
            });
            _this.activeTabName = "qa";

            // listen to column title change & set fields for DIMENSION exploration 
            //_this.watchTableFunc("qa");
          })
          this.userinput = "";
        } else {
          alert("Please upload at least one PDF file first!")
        }

      }
    },
    handleFileChange(file) {
      console.log("file changed", file);
      file.processed = false;

      // if file is not included in this.fileList, add it
      if (!this.fileList.includes(file)) {
        this.fileList.push(file);
        // upload files
        let formData = new FormData();
        formData.append('file', file.raw);
        service.upload(formData, () => {
        })
      }

      // add row to tabulator
      this.pdfTable.addRow(file, false)
        .then(function (row) {
          //row - the row component for the row updated or added
          row.select();
          //run code after data has been updated
        })
        .catch(function (error) {
          //handle error updating data
          console.log("error adding row", error)
        });
      this.uploadFiles();

    },
    downloadDB() {
      console.log("download db");
      this.dbTable.download("csv", "data.csv");
      // record user interactions
      this.userInteractions.endTime = new Date().getTime();
      this.userInteractions.messages = this.messages;

      console.log("user interactions:", this.userInteractions);
    },
    QAclearFilter() {
      this.qaSelectedField = "";
      this.qaSelectedOpt = "";
      this.qaFilterVal = "";
      this.qaTable.clearFilter();
      // record user interactions
      this.userInteractions.table_interactions.push({
        "time": new Date().getTime(),
        "type": "clear_filter",
        "view": "current_results",
        "target": "tablecell"
      })
    },
    clearFilter() {
      this.selectedField = "";
      this.selectedOpt = "";
      this.filterVal = "";
      this.dbTable.clearFilter();
      // record user interactions
      this.userInteractions.table_interactions.push({
        "time": new Date().getTime(),
        "type": "clear_filter",
        "view": "db",
        "target": "tablecell"
      })
    },
    addToDB() {
      const _this = this;
      console.log("add to db");
      // console.log("db table data: ", this.dbTable.getData());
      // console.log("qa table data: ", this.qaTable.getData());

      let newDBData = utils.joinArrays(this.qaTable.getData(), this.dbTable.getData(), ["pdf_file"], "outer")
      console.log("(new) db table data: ", newDBData);
      this.dbTable = new Tabulator("#db-table", {
        data: newDBData,
        autoColumns: true,
        autoColumnsDefinitions: function (definitions) {
          definitions.forEach((column) => {
            // console.log("column", column);
            // column.headerFilter = true; // add header filter to every column
            column.editor = true;
            column.cellClick = function (e, cell) {
              let filename = cell.getValue();
              let idx = _this.fileList.findIndex((file) => {
                return file.name == filename;
              })
              _this.selectedFile = _this.fileList[idx].raw;
            }
            // column.editableTitle = true;
            // column.headerWordWrap = true
          });

          return definitions;
        },
        columnDefaults: {
          maxWidth: 250, //maximum column width of 300px for all columns
        },
      });


      //_this.watchTableFunc("db");
      _this.fields = Object.keys(newDBData[0]).map(field => {
        return { "value": field }
      })

      _this.activeTabName = "db";

    },
    // display PDF info.
    displayPDFTable(filename) {
      console.log("displayPDFTable", filename);

      service.extract_table_from_pdf([{ "name": filename }], this.openai_key, tables => {
        console.log("extract_table_from_pdf: ", tables[0], typeof tables[0], typeof tables);
        this.tableLists = tables[0];
      });
    },
    displayPDFMeta(filename) {
      console.log("displayPDFMeta", filename);

      service.extract_meta_from_pdf([{ "name": filename }], this.openai_key, meta => {
        // console.log("extract_meta_from_pdf: ", meta[0]);
        this.metaInfo = meta[0];
      });
    },
    displayPDFFigure(filename) {
      console.log("displayPDFFigure", filename);

      service.extract_figure_from_pdf([{ "name": filename }], this.openai_key, figures => {
        console.log("extract_figure_from_pdf: ", figures[0]);
        this.figLists = figures[0];
      });
    },
    // left panel toggling
    toggleSidebar() {
      this.isSidebarOpen = !this.isSidebarOpen;
      this.isTooltipVisible = false; // Hide tooltip on toggle
    },
    handleMenuOption(option) {
      if (option === 'summarize') {
        this.dialogVisible = true;
        this.summary = "";
        if (this.tobeSummarized.length > 0) {
          service.summarize(this.tobeSummarized, summary => {
            console.log("summarize", summary);
            this.summary = summary;
            this.summaryLoading = false;
          })
        }
      }
    },

    // handle cell right click context menu options
    handleTableRightClickMenuOptions(cell) {
      console.log("handleTableRightClickMenuOptions: cell.getRow().getPosition(): ", cell.getRow().getPosition() - 1)
      let pdf_file = cell.getData().pdf_file;
      let fidx = this.fileList.findIndex((file) => {
        return file.name == pdf_file;
      });
      this.selectedFile = this.fileList[fidx].raw;

    },
    handleOpenContext(e, row) {
      // qa context checking
      console.log("right click in qa table", e, row.getData());
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      // Adjust if the context-display would go off the right edge of the viewport
      let left = e.clientX, top = e.clientY, width = 750, height = 300;
      if (left + width > viewportWidth) {
        left -= (left + width) - viewportWidth;
      }
      // Adjust if the context-display would go off the bottom edge of the viewport
      if (top + height > viewportHeight) {
        top -= (top + height) - viewportHeight;
      }

      // set current page to be 1
      this.currentPage = 1

      this.contexts = this.qaContexts[this.messages[this.messages.length - 2]][row.getData().pdf_file];
      this.qaCntxtStyle = {
        position: 'absolute',
        top: `${top}px`,
        left: `${left}px`
      };
      this.qaContxtVisible = true;
      ////////////////////////////////////////
      // get the data from the row
      ////////////////////////////////////////
      let currTokens = [];
      Object.keys(row.getData()).map(
        key => {
          if (key != "pdf_file" && key != "processed") {
            let val = row.getData()[key];
            if (typeof val === 'string') {  // Ensure the value is a string before tokenizing
              myTokenizer.tokenize(val).map((token) => {
                if (token.tag != 'punctuation' && (removeStopwords([token.value]).length > 0) && (token.value.length >= 2)) {
                  currTokens.push(token.value);
                }
              });
            }
          }
        });
      this.currTokens = currTokens;
      console.log("current tokens for the clicked row: ", currTokens);
      ////////////////////////////////////////
    },
    updateUserColumnInput(text) {
      this.userinput = text
    },
    updateQaCntxtStyle(newStyle) {
      this.qaCntxtStyle = { ...newStyle };
    }
  },
  mounted() {
    const _this = this;
    this.loadFiles();

    // pdf table
    this.pdfTable = new Tabulator("#pdf-table", {
      columns: [
        {
          formatter: "rowSelection", titleFormatter: "rowSelection", hozAlign: "center", headerSort: false, cellClick: function (e, cell) {
            cell.getRow().toggleSelect();
          }
        },
        {
          title: "Name", field: "name", width: 240, cellClick: function (e, cell) {
            let filename = cell.getValue();
            console.log("cellClick filename", filename);
            let idx = _this.fileList.findIndex((file) => {
              return file.name == filename;
            })
            _this.selectedFile = _this.fileList[idx].raw;
          }
        },
        {
          title: "Delete", formatter: "buttonCross", width: 100, hozAlign: "center", cellClick: function (e, cell) {
            cell.getRow().delete();
          }
        },
      ],
    });

    // db table
    this.dbTable = utils.buildTable('db-table', { contentEdittable: true, colTitleEditable: true });

    // qa table
    this.qaTable = utils.buildTable('qa-table', { contentEdittable: true, colTitleEditable: true });
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

.messages {
  padding: 10px;
  font-size: 13px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}


.message-row {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.message-container {
  background-color: #f0f0f0;
  padding: 10px 15px;
  border-radius: 15px;
  max-width: 80%;
  word-break: break-word;
  margin-left: 10px;
  /* Space between icon and message */
}

.user-message {
  background-color: #e0e0ff;
  /* Different background for user messages */
}

.ai-message {
  background-color: #f0f0f0;
  /* AI messages */
}

.user-icon,
.ai-icon {
  font-size: 24px;
  /* Icon size */
}

.header-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #545c64;
  padding: 10px;
}


.chat-container {
  /* max-width: 600px; */
  margin: auto;
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  background-color: #f9f9f9;
  font-family: sans-serif;
}

textarea {
  width: 100%;
  padding: 12px 20px;
  box-sizing: border-box;
  border: none;
  border-top: 1px solid #ddd;
  font-size: 14px;
  color: #333;
}

input[type="text"]::placeholder {
  color: #aaa;
}

input[type="text"]:focus {
  outline: none;
}


.el-row {
  padding: 5px;
}

.el-container,
.el-col,
.el-aside {
  padding-left: 2px;
}

.header-web {
  font-size: 20px;
  /* Adjusted for better visibility */
  font-weight: 500;
  /* Semi-bold font-weight */
  letter-spacing: 0.5px;
  /* Adding some letter-spacing for elegance */
  user-select: none;
  /* Prevents text from being selectable */
  display: flex;
  /* Ensures proper alignment */
  align-items: center;
  /* Centers the content vertically */
  justify-content: space-between;
  /* Maximizes the space between items */
}

.el-header {
  color: black;
  line-height: 50px;
  /* Vertically centers the text */
  padding-left: 5px !important;
  font-family: 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
  /* Modern font-stack */
}

.el-aside {
  color: #333;
}

.el-col {
  border-radius: 4px;
}


/* General row styling */
.tablefilter-row {
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 10px;
  /* consistent spacing between elements */
}

/* Style for the select and input elements */
.tablefilter-select,
.tablefilter-input {
  border-radius: 4px;
  border: 1px solid #dcdfe6;
}

.tablefilter-select {
  width: 200px;
  /* Adjusted width for better appearance */
}

.tablefilter-input {
  width: 200px;
  /* Keeping your original width */
}

/* Button styling */
.tablefilter-button {
  background-color: #409EFF;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 5px 15px;
  cursor: pointer;
}

.tablefilter:hover {
  background-color: #66b1ff;
}

.block {
  margin: 15px;
  padding: 10px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.demonstration {
  font-size: 13px;
}


.el-switch {
  margin-right: 5px;
}

.toolbar {
  display: flex;
  /* justify-content: space-between; */
  margin-bottom: 10px;
  /* justify-content: center; */
  align-items: center;
  flex-wrap: wrap;
  /* Allows items to wrap to the next line */
}

.toolbar button {
  padding: 5px 10px;
  margin-right: 5px;
  background-color: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 3px;
  cursor: pointer;
}

.toolbar button:hover {
  background-color: #e8e8e8;
}

.el-checkbox {
  margin-right: 5px !important;
  margin-left: 5px !important;
}

/* left side bar */
.content-col {
  position: relative;
  /* Needed for absolute positioning of the button */
}

.toggle-button-container {
  position: absolute;
  top: 30%;
  /* Center vertically */
  left: 15px;
  /* Align to the left edge */
  transform: translateY(-50%) translateX(-100%);
  /* Adjust for exact centering and move it to the left */
  z-index: 10;
  /* Ensure it's above other content */
}

.sidebar-toggle-button {
  background-color: transparent;
  /* Primary color */
  color: white;
  /* Text and icon color */
  border: none;
  border-radius: 3px;
  padding: 2px 0px;
  cursor: pointer;
  transition: background-color 0.3s, transform 0.2s, height 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sidebar-toggle-button:hover {
  color: lightgray;
  /* Darker shade of gray for hover */
  transform: scale(1.4);
  /* Slight increase in size */
}

.sidebar-toggle-button i {
  color: lightgray;
  /* Dim gray color for the icon */
  font-size: 1.5em;
  /* Adjust icon size */
  font-weight: bold;
  border: 2px;
  text-shadow: 1px 0 lightgray, -1px 0 lightgray, 0 1px lightgray, 0 -1px lightgray;
  /* Faux-thickening effect */
}

.el-button i {
  font-size: 1.2em;
}

/* context menu */
.context-menu {
  border: 1px solid #ccc;
  background-color: white;
  border-radius: 5px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
  font-size: 12px;
}

.context-menu ul {
  list-style: none;
  margin: 0;
  padding: 2px;
}

.context-menu ul li {
  padding: 8px 15px;
  cursor: pointer;
  border-bottom: 1px solid #eee;
}

.context-menu ul li:hover {
  background-color: lightgray;
}

/* context */
.context-display {
  margin: auto;
  width: 750px;
  overflow: scroll;
  z-index: 1000;
}

.context-text {
  white-space: pre-wrap;
  /* Preserves spaces and line breaks */
  font-size: 13px;
  height: 250px;
  overflow: scroll;
}

/* tabulator FONT */
.tabulator {
  font-size: 13px;
}

.contextTable {
  font-size: 12px;
}

.highlight {
  background-color: #ff0;
}
</style>
