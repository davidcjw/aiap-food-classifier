var app = new Vue({
    el: '#app',
    data() {
        return {
            selectedFile: null,
            imageURL: null,
            results: "",
            errorMessage: ""
        }
    },
    methods: {
        onFileChanged(event) {
            this.selectedFile = event.target.files[0];
            this.imageURL = URL.createObjectURL(this.selectedFile);
            this.results = "";
        },
        onUpload() {
            let vm = this;
            const formData = new FormData()
            formData.append('file', this.selectedFile, this.selectedFile.name)
            axios.post('predict', formData)
                .then(function (response) {
                    vm.results = response.data;
                    console.log(response);
                })
                .catch(function (error) {
                    vm.errorMessage = error.response.data.error_message;
                    console.log(error.response.data);
                });
        }
    }
})
