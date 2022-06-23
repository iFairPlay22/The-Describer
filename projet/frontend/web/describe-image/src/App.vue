<template>
  <div id="app">
    <main-header :data="headerData" />

    <div id="body">
      <el-image
        style="
          width: 200px; 
          height: 200px;
          position: fixed;
          right: 0;
          top: 0;
          z-index: 100;
        "
        src="/images/logos/qr_code.png"
        fit="cover"
        class="small-image"
      />

      <big-text :data="title1Data" />
      <description-row :data="descriptionData" />
      <big-text :data="title2Data" />
      <ai-row :data="aiData" />
      <big-text :data="title3Data" />
      <services-row :data="servicesData" />
    </div>
  </div>
</template>


<script>
import MainHeader from './components/MainHeader.vue';
import DescriptionRow from './components/DescriptionRow.vue';
import AiRow from './components/AiRow.vue';
import ServicesRow from './components/ServicesRow.vue';
import BigText from './components/BigText.vue';

export default {
  name: 'App',
  components: { MainHeader, DescriptionRow, AiRow, ServicesRow, BigText },
  data() {
    return {
      headerData: {
        name: "The Describers",
        links: [
          {
            name: "Origine",
            id: "what"
          },
          {
            name: "L'IA",
            id: "ai"
          },
          {
            name: "Services",
            id: "services"
          }
        ]
      },
      title1Data: {
        id: "what",
        text: "Origine & Contexte",
      },
      descriptionData: {
        title: "Ecosystème d'aide aux personnes malvoyantes",
        text: "Un projet innovant pour un monde compris de tous ! 🙄",
        image: "/images/big/big_cat.jpg",
        groups: [
          {
            title: "Projet",
            text: "Création d'une intelligence artificielle capable de décrire une image et son contetxe"
          },
          {
            title: "Objectif",
            text: "Rendre accessible à tous la description d'image sur nos outils numériques"
          }
        ],
      },
      title2Data: {
        id: "ai",
        text: "Notre IA",
      },
      aiData: {
        strings: {
          title: "Comment ça marche ?",
          text: "Sélectionnez une des images ci-dessous pour obtenir la traduction associée. Vous pouvez aussi uploader vos images personnalisées en cliquant sur l'image de droite.",
          uploadButtonText: "Uploader une image",
          reloadButtonText: "Charger d'autres images",
          processingText: "Traitement en cours ...",
          errorText: "Une erreur est survenue...",
          alerts: {
            successMessage: "Description effectuée avec succès !",
            badFileMessage: "Fichier trop lourd, où format non pris en charge...",
            errorMessage: 'Une erreur inattendue est survenue ! 👀',
            pleaseWaitMessage: 'Veuillez attendre quelques secondes... 🥴'
          }
        },
        proposedImageProperties: {
          total: 6,
          width: 200,
          height: 150
        },
        backendApi: "https://www.loicfournier.fr",
        backendToken: "mbzy52iNo9bo6sGYXL6WGacc9LQz1mvv",
        userLocale: "en",
      },
      title3Data: {
        id: "services",
        text: "Nos services",
      },
      servicesData: {
        services: [
          {
            avatar: "/images/logos/mobile_app_logo.png",
            title: "Application mobile",
            description: "Prennez des photos, et obtenez la description ! Disponible sur Android & IOS ! ✨",
            button: "Télécharger",
            callback: () => { window.open("/downloads/the_describers_app.zip", '_blank') }
          },
          {
            avatar: "/images/logos/discord_logo.png",
            title: "Bot Discord",
            description: "Notre IA, directement accessible sur vos serveurs ! 💕",
            button: "Télécharger",
            callback: () => { window.open("https://discord.com/api/oauth2/authorize?client_id=978558654427041802&permissions=2048&scope=bot", '_blank') }
          },
          {
            avatar: "/images/logos/chrome_logo.png",
            title: "Extension Chrome",
            description: "Décrit automatiquement les images sur votre navigateur ! 👀",
            button: "Télécharger",
            callback: () => { window.open("/downloads/the_describers_chrome_ext.zip", '_blank') }
          }
        ]
      }
    };
  },
  created() {
    const userLanguage =
      navigator && navigator.languages && navigator.languages.length
        ? navigator.languages[0]
        : navigator.language;
    this.aiData.userLocale = userLanguage;
  }
}
</script>