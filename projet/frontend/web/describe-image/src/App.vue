<template>
  <div id="app">
    <main-header/>
    
    <div id="body">
      <big-text
        id="what"
        text="Origine & Contexte"
      />
      <description-row/>
      <big-text
        id="ai"
        text="Notre IA"
      />
      <ai-row/>
      <hr>
    </div>
  </div>
</template>


<script>
import MainHeader from './components/MainHeader.vue';
import DescriptionRow from './components/DescriptionRow.vue';
import AiRow from './components/AiRow.vue';
import BigText from './components/BigText.vue';

export default {
  name: 'App',
  components: { MainHeader, DescriptionRow, AiRow, BigText }
}
</script>