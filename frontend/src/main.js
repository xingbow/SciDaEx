import Vue from 'vue'
import App from './App.vue'

// ElementUI
import ElementUI from 'element-ui';
import locale from 'element-ui/lib/locale/lang/en'

import 'element-ui/lib/theme-chalk/index.css';
Vue.use(ElementUI, {locale});

import * as d3 from 'd3'
window.d3 = d3

import _ from 'lodash'
window._ = _

import $ from 'jquery'
window.$ = $

// fontawesome
require('../node_modules/@fortawesome/fontawesome-free/css/all.css');

import "../node_modules/tabulator-tables/dist/css/tabulator_semanticui.css"

import "../node_modules/hint.css/hint.css"

Vue.config.productionTip = false

function mountApp() {
  new Vue({
    render: h => h(App),
  }).$mount('#app')
}

if (window.adobeViewerLoaded) {
  mountApp();
} else {
  window.addEventListener('adobeViewerReady', mountApp);
}
