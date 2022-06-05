<template>
    <el-row class="normal-y-margin">
        <el-col :span="24" :md="12" class="merge-container-left">
            <el-row style="margin: 20px 0">
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
                        <el-col :xs="24" :sm="12">
                            <el-card class="rounded-borders-except-bl small-margin merge-cards-left">
                                <div class="title">Projet</div>
                                <div class="text">Création d'une intelligence artificielle capable de décrire des images</div>
                            </el-card>
                        </el-col>
                        <el-col :xs="24" :sm="12">
                            <el-card class="rounded-borders-except-tr small-margin merge-cards-right">
                                <div class="title">Objectif</div>
                                <div class="text">Obtenir plus facilement la description d'image sur nos outils numériques</div>
                            </el-card>
                        </el-col>
                    </el-row>
                </el-col>
            </el-row>
        </el-col>
        <el-col :xs="24" :sm="12" class="merge-container-right">
            <el-image
                class="big-home-image"
                :src="url"
                fit="cover">
            </el-image>
        </el-col>
    </el-row>
</template>

<script>
export default {
  props: {
  },
    name: "DescriptionRow",
  data() {
    return {
      url: "https://voyage-onirique.com/wp-content/uploads/2021/10/2651.jpg"
    }
  }
}
</script>

<style>

</style>