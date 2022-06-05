<template>
    <el-row class="normal-y-margin">
        <el-col :md="12" :span="24">
            <el-row>
                <el-col :span="24">
                    <cite class="text-center small-y-margin"> 
                        <el-icon class="el-icon-d-arrow-left small-x-margin"/>
                        <span class="big-text"> {{ formattedCurrentDescription }} </span>
                        <el-icon class="el-icon-d-arrow-right small-x-margin"/>
                    </cite>
                </el-col>
            </el-row>
            <el-row>
                <el-col :span="24">
                    <el-image
                        class="big-home-image"
                        :src="currentImage"
                        fit="cover">
                    </el-image>
                </el-col>
            </el-row>
        </el-col>

        <el-col :md="12" :span="24">
            <el-row>
                <el-col :span="24">
                    <el-card class="without-borders">
                        <div class="title">Comment ça marche ?</div>
                        <div class="text">
                            Sélectionnez une des images ci-dessous pour obtenir la traduction associée. 
                            Vous pouvez aussi uploader vos images personnalisées en cliquant sur l'image de droite.
                        </div>
                        <div class="button-bar">
            
                            <el-upload
                                action="#"
                                :auto-upload="false"
                                :limit="1"
                                :file-list="uploadedFiles"
                                :on-change="updateDescriptionByFile"
                            >
                                <el-button type="primary" class="very-small-margin">
                                    <i class="el-icon-upload"></i>
                                    Uploader une image
                                </el-button>
                            </el-upload>

                            <el-button type="primary" @click="generateRandomImages" class="very-small-margin">
                                <i class="el-icon-plus"></i>
                                Charger d'autres images
                            </el-button>
                        </div>
                    </el-card>
                </el-col>
            </el-row>
            <el-row>
                <el-col :span="24">
                    <el-card class="without-borders">
                        <div 
                            class="small-image-container">
                            <el-image
                                :style="'width: ' + proposedImageProperties.width + 'px;' + 'height:' + proposedImageProperties.height + 'px;'"
                                v-for="(proposedImage, i) in proposedImages" 
                                :key="i"
                                :src="proposedImage"
                                fit="cover"
                                class="small-image"
                                @click="updateDescriptionByUrl(proposedImage)"
                            />
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
  data() {
    return {
      currentImage: "https://havingfun.fr/wp-content/uploads/2017/05/surf-wallpaper-3.jpg",
      currentDescription: "un homme chevauchant une planche de surf sur une vague dans l'océan.",
      proposedImageProperties: {
          total: 6,
          width: 200,
          height: 150
      },
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
            
            while (imgs.length != this.proposedImageProperties.total) {
                const id = this.randomIntFromInterval(1, 63);
                if (!imgs.includes(id))
                    imgs.push(id);
            }

            this.proposedImages = imgs.map(id => `${window.location.origin}/images/${id}.jpg`);
        },
        randomIntFromInterval(min, max) {  
            return Math.floor(Math.random() * (max - min + 1) + min)
        },
        updateImage(image) {
            this.currentImage = image;
            this.currentDescription = "Traitement en cours ...";
        },
        updateDescriptionByUrl(url) {
            
            this.updateImage(url);

            var myHeaders = new Headers();
            myHeaders.append("Content-Type", "application/json");

            var raw = JSON.stringify({ "file": this.currentImage });

            var requestOptions = {
                method: 'POST',
                headers: myHeaders,
                body: raw,
                redirect: 'follow'
            };

            this.apiRequest("http://217.160.10.8:80/iadecode/from_url", requestOptions);
        },
        updateDescriptionByFile(image) {
            
            if (!image) return
            image = image.raw
            this.uploadedFiles = [];
            
            this.updateImage(URL.createObjectURL(image));

            var formdata = new FormData();
            formdata.append("file", image);

                var requestOptions = {
                method: 'POST',
                body: formdata,
                redirect: 'follow'
            };

            this.apiRequest("http://217.160.10.8:80/iadecode/from_file", requestOptions);
        },
        apiRequest(url, options) {
          
            fetch(url, options)
                .then(response => {
                    if (response.status == 200) {
                        response.json()
                            .then(({ message }) => {
                                this.currentDescription = message;    
                                this.$message.success({ message: "Description effectuée avec succès : \"" + this.formattedCurrentDescription + "\".", center: true, showClose: true, duration: 10000 });
                            })
                            .catch(() => {
                                this.$message.success({ message: "Format de fichiers non pris en charge...", center: true, showClose: true, duration: 10000 });
                                this.currentDescription = "Une erreur est survenue...";
                            });
                    }
                })
                .catch(() => {
                    this.$message.error({ message: 'Une erreur inattendue est survenue ! 👀', center: true, showClose: true, duration: 10000 });
                });
        }
  },
}
</script>

<style>

</style>