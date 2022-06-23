<template>
    <el-row class="normal-y-margin">
        <el-col :md="12" :span="24">
            <el-row>
                <el-col :span="24">
                    <cite class="text-center small-y-margin">
                        <el-icon class="el-icon-d-arrow-left small-x-margin" />
                        <span class="big-text"> {{ formattedCurrentDescription }} </span>
                        <el-icon class="el-icon-d-arrow-right small-x-margin" />
                    </cite>
                </el-col>
            </el-row>
            <el-row>
                <el-col :span="24">
                    <el-image class="big-home-image" :src="currentImage" fit="cover">
                    </el-image>
                </el-col>
            </el-row>
        </el-col>

        <el-col :md="12" :span="24">
            <el-row>
                <el-col :span="24">
                    <el-card class="without-borders">
                        <div class="title">{{ data.strings.title }}</div>
                        <div class="text">{{ data.strings.text }}</div>
                        <div class="button-bar">

                            <el-upload action="#" :auto-upload="false" :limit="1" :file-list="uploadedFiles"
                                :on-change="updateDescriptionByFile">
                                <el-button type="primary" class="very-small-margin" :disabled="!canClick">
                                    <i class="el-icon-upload"></i>
                                    {{ data.strings.uploadButtonText }}
                                </el-button>
                            </el-upload>

                            <el-button type="primary" @click="generateRandomImages" class="very-small-margin">
                                <i class="el-icon-plus"></i>
                                {{ data.strings.reloadButtonText }}
                            </el-button>
                        </div>
                    </el-card>
                </el-col>
            </el-row>
            <el-row>
                <el-col :span="24">
                    <el-card class="without-borders">
                        <div class="small-image-container">
                            <el-image
                                :style="'width: ' + data.proposedImageProperties.width + 'px;' + 'height:' + data.proposedImageProperties.height + 'px;'"
                                v-for="(proposedImage, i) in proposedImages" :key="i" :src="proposedImage" fit="cover"
                                :class='"small-image " + (canClick ? "small-image-clickable" : "")' @click="updateDescriptionByUrl(proposedImage)" />
                        </div>
                    </el-card>
                </el-col>
            </el-row>
        </el-col>
    </el-row>
</template>

<script>
export default {
    name: "DescriptionRow",
    props: {
        data: {
            strings: {
                title: {
                    type: String,
                    default: "",
                },
                text: {
                    type: String,
                    default: "",
                },
                uploadButtonText: {
                    type: String,
                    default: "",
                },
                reloadButtonText: {
                    type: String,
                    default: "",
                },
                processingText: {
                    type: String,
                    default: "",
                },
                errorText: {
                    type: String,
                    default: "",
                },
                alerts: {
                    successMessage: {
                        type: String,
                        default: "",
                    },
                    badFileMessage: {
                        type: String,
                        default: "",
                    },
                    errorMessage: {
                        type: String,
                        default: "",
                    },
                    pleaseWaitMessage: {
                        type: String,
                        default: "",
                    },
                }

            },
            proposedImageProperties: {
                total: {
                    type: Number,
                    default: 6,
                },
                width: {
                    type: Number,
                    default: 200,
                },
                height: {
                    type: Number,
                    default: 150,
                },
            },
            backendApi: {
                type: String,
                default: "",
            },
            backendToken: {
                type: String,
                default: ""
            },
            userLocale: {
                type: String,
                default: "",
            },
        }
    },
    data() {
        return {
            canClick: true,
            currentImage: "/images/big/big_surf.jpg",
            currentDescription: "Un homme chevauchant une vague sur une planche de surf.",
            proposedImages: [],
            uploadedFiles: []
        }
    },
    created() {
        this.generateRandomImages();
    },
    computed: {
        formattedCurrentDescription() {
            if (!this.currentDescription)
                return "";

            return this.currentDescription.charAt(0).toUpperCase() + this.currentDescription.slice(1);
        }
    },
    methods: {
        generateRandomImages() {
            let imgs = [];

            while (imgs.length != this.data.proposedImageProperties.total) {
                const id = this.randomIntFromInterval(1, 151);
                if (!imgs.includes(id))
                    imgs.push(id);
            }

            this.proposedImages = imgs.map(id => `${window.location.origin}/images/proposed/${id}.jpg`);
        },
        randomIntFromInterval(min, max) {
            return Math.floor(Math.random() * (max - min + 1) + min)
        },
        updateImage(image) {
            this.currentImage = image;
            this.currentDescription = this.data.strings.processingText;
        },
        canContinue() {
            if (!this.canClick) {
                this.$message.warning({ message: this.data.strings.alerts.pleaseWaitMessage, center: true, showClose: true, duration: 10000 });
                return false;
            }
            this.canClick = false;
            return true;
        },
        updateDescriptionByUrl(url) {

            if (!this.canContinue())
                return

            this.updateImage(url);

            var myHeaders = new Headers();
            myHeaders.append("Content-Type", "application/json");
            myHeaders.append("Authorization", "Bearer " + this.data.backendToken);

            var raw = JSON.stringify({ "file": this.currentImage });

            var requestOptions = {
                method: 'POST',
                headers: myHeaders,
                body: raw,
                redirect: 'follow'
            };

            this.apiRequest(this.data.backendApi + "/iadecode/from_url/" + this.data.userLocale, requestOptions);
        },
        updateDescriptionByFile(image) {

            if (!this.canContinue())
                return

            if (!image) return
            image = image.raw
            this.uploadedFiles = [];

            this.updateImage(URL.createObjectURL(image));

            var myHeaders = new Headers();
            myHeaders.append("Authorization", "Bearer " + this.data.backendToken);

            var formdata = new FormData();
            formdata.append("file", image);

            var requestOptions = {
                method: 'POST',
                headers: myHeaders,
                body: formdata,
                redirect: 'follow'
            };

            this.apiRequest(this.data.backendApi + "/iadecode/from_file/" + this.data.userLocale, requestOptions);
        },
        apiRequest(url, options) {

            fetch(url, options)
                .then(response => {
                    if (response.status == 200) {
                        response.json()
                            .then(({ message }) => {
                                this.currentDescription = message;
                                this.$message.success({ message: this.data.strings.alerts.successMessage, center: true, showClose: true, duration: 10000 });
                            })
                            .catch(() => {
                                this.$message.success({ message: this.data.strings.alerts.badFileMessage, center: true, showClose: true, duration: 10000 });
                                this.currentDescription = this.data.strings.errorText;
                            })
                    }
                })
                .catch(() => {
                    this.$message.error({ message: this.data.strings.alerts.errorMessage, center: true, showClose: true, duration: 10000 });
                })
                .finally(() => {
                    this.canClick = true;
                });
        }
    },
}
</script>

<style>
</style>