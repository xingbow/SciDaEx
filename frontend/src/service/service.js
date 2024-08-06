import axios from 'axios'
const GET_REQUEST = 'get'
const POST_REQUEST = 'post'
const backendPort = 5010
const dataServerUrl = `http://127.0.0.1:${backendPort}/api`
const yourAdobeClientId = "please enter your own adobe client id"
const loginpassword = "llm2024!"
const alertTh = .25

function request(url, params, type, callback) {
    let func
    if (type === GET_REQUEST) {
        func = axios.get
    } else if (type === POST_REQUEST) {
        func = axios.post
    }

    func(url, params).then((response) => {
            if (response.status === 200) {
                callback(response["data"])
            } else {
                console.error(response) /* eslint-disable-line */
            }
        })
        .catch((error) => {
            console.error(error) /* eslint-disable-line */
        })
}

function getFiles(callback) {
    let url = `${dataServerUrl}/files`
    request(url, null, GET_REQUEST, callback)
}

function upload(formdata, callback) {
    let url = `${dataServerUrl}/upload`
    const params = formdata
    request(url, params, POST_REQUEST, callback)
}

// run qa on users' question
function run_qa(q, fileLists, callback) {
    // console.log("dataset: ", dataset);
    let url = `${dataServerUrl}/qa`   
    const params = {
        "question": q,
        "filenames": fileLists
    }
    request(url, params, POST_REQUEST, callback)
}

// extract tables from pdf
function extract_table_from_pdf(fileLists, openai_key, callback) {
    let url = `${dataServerUrl}/extract_table_from_pdf`
    const params = {
        "filenames": fileLists,
        "openai_api_key": openai_key
    }
    request(url, params, POST_REQUEST, callback)
}

// extract meta from pdf
function extract_meta_from_pdf(fileLists, openai_key, callback) {
    let url = `${dataServerUrl}/extract_meta_from_pdf`
    const params = {
        "filenames": fileLists,
        "openai_api_key": openai_key
    }
    request(url, params, POST_REQUEST, callback)
}

// extract figure from pdf
function extract_figure_from_pdf(fileLists, openai_key, callback) {
    let url = `${dataServerUrl}/extract_figure_from_pdf`
    const params = {
        "filenames": fileLists,
        "openai_api_key": openai_key
    }
    request(url, params, POST_REQUEST, callback)
}

// summarize pdf
function summarize(fileList, callback) {
    let url = `${dataServerUrl}/summarize`
    const params = fileList
    request(url, params, POST_REQUEST, callback)
}

// get relevance score
function getRelevanceScore(params, callback) {
    let url = `${dataServerUrl}/get_confidence_scores`
    request(url, params, POST_REQUEST, callback)
}

export default {
    getFiles,
    yourAdobeClientId,
    loginpassword,
    alertTh,
    dataServerUrl,
    upload,
    run_qa,
    extract_table_from_pdf,
    extract_meta_from_pdf,
    extract_figure_from_pdf,
    summarize,
    getRelevanceScore
}