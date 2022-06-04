<template>
  <div id="app">
    <main-header/>
    
    <div id="body">
      <el-row>
        <el-col :span="12">
          <el-row>
            <el-col :span="24">
              <el-card class="without-borders">
                <div class="title">Ecosystème d'aide aux<br/> personnes mal-voyantes</div>
                <div class="text">Un projet innovant pour un monde meilleur</div>
              </el-card>
            </el-col>
          </el-row>
          <el-row>
            <el-col :span="24">
              <el-row>
                <el-col :span="12">
                  <el-card class="rounded-borders-except-bl small-margin merge-cards-left">
                    <div class="title">Projet</div>
                    <div class="text">Création d'une intelligence artificielle capable de décrire des images</div>
                  </el-card>
                </el-col>
                <el-col :span="12">
                  <el-card class="rounded-borders-except-tr small-margin merge-cards-right">
                    <div class="title">Objectif</div>
                    <div class="text">Obtenir plus facilement la description d'image sur nos outils numériques</div>
                  </el-card>
                </el-col>
              </el-row>
            </el-col>
          </el-row>
        </el-col>
        <el-col :span="12">
          <div class="big-home-image-container">
            <el-image
              class="big-home-image"
              :src="url"
              fit="fill">
            </el-image>
          </div>
        </el-col>
      </el-row>
    </div>
  </div>
</template>


<script>
import MainHeader from './components/MainHeader.vue';

export default {
  name: 'App',
  components: { MainHeader },
  data() {
    return {
      url: "https://media-public.canva.com/tEBUk/MAEfz4tEBUk/1/s.jpg"
    }
  }
}
</script>