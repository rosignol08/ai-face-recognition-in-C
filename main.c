#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <linux/videodev2.h>
#include <fcntl.h>   // Pour open()
#include <unistd.h>  // Pour close(), read(), write()
#include <sys/ioctl.h>  // Pour ioctl()
#include <sys/mman.h>  // Pour mmap(), munmap()
#include <time.h>  // Pour srand(), time()
#define WEIGHT_MAX 10
#define WEIGHT_MIN -10

#include <dirent.h> // Pour opendir, readdir, closedir
#include <sys/stat.h> // Pour S_ISREG et pour DT_REG

#include <errno.h> // Pour errno
//#include <dirent.h> // Pour struct dirent 

//pour les images jpg
#include <jpeglib.h>
#include <jerror.h>


#define MAX_CLASS_NAME_LENGTH 256
#define MAX_CLASSES 5749


// Structure pour stocker la classe et son index en gros les nom et prenom des gens sur les photos
struct ClassMapping {
    char name[MAX_CLASS_NAME_LENGTH];
    int index;
};

// Fonction pour trouver ou ajouter une nouvelle classe dans le dictionnaire
int findOrAddClass(struct ClassMapping* classMappings, int* numClasses, const char* name) {
    for (int i = 0; i < *numClasses; i++) {
        if (strcmp(classMappings[i].name, name) == 0) {
            return classMappings[i].index; // Classe existante
        }
    }

    // Ajouter une nouvelle classe
    if (*numClasses >= MAX_CLASSES) {
        fprintf(stderr, "Nombre maximum de classes atteint\n");
        return -1;
    }

    strcpy(classMappings[*numClasses].name, name);
    classMappings[*numClasses].index = *numClasses;
    return (*numClasses)++;
}

//structure du neurone
struct Neuron{
    double weights;
    double bias;
    double output;
    double error;
};

struct Layer {
    struct Neuron* neurons;
    int numNeurons;
};

struct Network {
    struct Layer* layers;
    int numLayers;
};

void printNeuronWeights(struct Network* network, int layerIndex, int neuronIndex);

/*
    neuron->weights = (double*)malloc(numInputs * sizeof(double));
    if (neuron->weights == NULL) {
        perror("Erreur d'allocation de mémoire pour les poids");
        return;
    }
*/

//inintialisation des poids
void initializeWeights(struct Neuron* neuron) {
    // Initialisez le poids avec des valeurs aléatoires, par exemple entre -1 et 1
    neuron->weights = ((double)rand() / RAND_MAX) * 2 - 1; // Valeur entre -1 et 1
    printf("Poids: %f\n", neuron->weights);
}



//fonction d'activation
double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

//pour calculer les sorties (utilisée dans la fonction feedForwardNetwork)
void feedForward(struct Neuron* neuron, int n, double* inputs){
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += neuron->weights * inputs[i];
    }
    sum += neuron->bias;
    neuron->output = sigmoid(sum);
    return;
}

//pour calculer la sortie du reseau en fonction des couches
void feedForwardNetwork(struct Network* network, double* inputs) {
    double* currentInputs = inputs;
    for (int i = 0; i < network->numLayers; i++) {
        struct Layer* layer = &network->layers[i];
        double* newInputs = (double*)malloc(layer->numNeurons * sizeof(double));
        for (int j = 0; j < layer->numNeurons; j++) {
            feedForward(&layer->neurons[j], network->layers[i == 0 ? 0 : i-1].numNeurons, currentInputs);
            newInputs[j] = layer->neurons[j].output;
        }
        currentInputs = newInputs;
    }
}

//ajuster les poids en fonction de l'erreur (utilisée dans la fonction backpropagationNetwork)
void backpropagation(struct Neuron* neuron, int n, double* inputs, double learningRate){
    double gradient = neuron->output * (1 - neuron->output) * neuron->error;
    for (int i = 0; i < n; i++) {
        neuron->weights += learningRate * gradient * inputs[i];
    }
    neuron->bias += learningRate * gradient;
    return;
}

//fonction de cout pour calculer l'erreur
double cout(struct Network* network, double* expectedOutputs){
    double error = 0;
    struct Layer* outputLayer = &network->layers[network->numLayers - 1];
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        error += pow(expectedOutputs[i] - outputLayer->neurons[i].output, 2);
    }
    return error;
}


//ajuster les poids en fonction de l'erreur
void backpropagationNetwork(struct Network* network, double* expectedOutputs, double learningRate) {
    // Calculer l'erreur pour la dernière couche
    struct Layer* outputLayer = &network->layers[network->numLayers - 1];
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        outputLayer->neurons[i].error = expectedOutputs[i] - outputLayer->neurons[i].output;
    }

    // Propager l'erreur en arrière
    for (int i = network->numLayers - 2; i >= 0; i--) {
        struct Layer* currentLayer = &network->layers[i];
        struct Layer* nextLayer = &network->layers[i + 1];
        for (int j = 0; j < currentLayer->numNeurons; j++) {
            double errorSum = 0;
            for (int k = 0; k < nextLayer->numNeurons; k++) {
                errorSum += nextLayer->neurons[k].weights * nextLayer->neurons[k].error;
            }
            currentLayer->neurons[j].error = errorSum * currentLayer->neurons[j].output * (1 - currentLayer->neurons[j].output);
        }
    }

    // Ajuster les poids
    for (int i = 0; i < network->numLayers; i++) {
        struct Layer* layer = &network->layers[i];
        for (int j = 0; j < layer->numNeurons; j++) {
            backpropagation(&layer->neurons[j], i == 0 ? 0 : network->layers[i-1].numNeurons, i == 0 ? expectedOutputs : &(network->layers[i-1].neurons->output), learningRate);
        }
    }
}

// fonction pour prédire les sorties
void predict(struct Network* network, double* input, double* output) {
    feedForwardNetwork(network, input);
    // Copier les sorties du dernier layer
    struct Layer* outputLayer = &network->layers[network->numLayers - 1];
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        output[i] = outputLayer->neurons[i].output;
    }
}

//fonction pour calculer la précision
double calculateAccuracy(struct Network* network, double** validationInputs, double** validationOutputs, int numValidationSamples, int numOutputs) {
    int correctPredictions = 0;
    double* predictedOutput = (double*)malloc(numOutputs * sizeof(double));
    
    for (int i = 0; i < numValidationSamples; i++) {
        predict(network, validationInputs[i], predictedOutput);
        int isCorrect = 1;
        for (int j = 0; j < numOutputs; j++) {
            if (round(predictedOutput[j]) != validationOutputs[i][j]) {
                isCorrect = 0;
                break;
            }
        }
        if (isCorrect) {
            correctPredictions++;
        }
    }
    free(predictedOutput);
    return (double)correctPredictions / numValidationSamples * 100.0;
}

//on va utiliser le maxPooling pour réduire la taille du nombre de neurones nécessaires
void maxPooling(double* input, double* output, int inputWidth, int inputHeight, int poolSize, int stride) {
    int outputWidth = (inputWidth - poolSize) / stride + 1;
    int outputHeight = (inputHeight - poolSize) / stride + 1;

    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            double max = -__DBL_MAX__;
            for (int m = 0; m < poolSize; m++) {
                for (int n = 0; n < poolSize; n++) {
                    int inputX = j * stride + n;
                    int inputY = i * stride + m;
                    double value = input[inputY * inputWidth + inputX];
                    if (value > max) {
                        max = value;
                    }
                }
            }
            output[i * outputWidth + j] = max;
        }
    }
}

//initialiser le réseau
void initializeNetwork(struct Network* network, int height, int width, int numOutputs) {
    network->numLayers = 3; // Nombre de couches
    network->layers = (struct Layer*)malloc(network->numLayers * sizeof(struct Layer));

    if (network->layers == NULL) {
        perror("Erreur d'allocation de mémoire pour les couches du réseau");
        exit(EXIT_FAILURE);
    }

    // Dimensions après pooling
    int poolSize = 2;
    int stride = 2;
    int pooledWidth = (width - poolSize) / stride + 1;
    int pooledHeight = (height - poolSize) / stride + 1;
    int numInputs = pooledWidth * pooledHeight;

    // Exemple de configuration du nombre de neurones par couche
    int numNeurons[] = {numInputs, 830, numOutputs};
    for (int i = 0; i < network->numLayers; i++) {
        printf("Layer %d: %d neurones\n", i, numNeurons[i]);
    }
    for (int i = 0; i < network->numLayers; i++) {
        network->layers[i].numNeurons = numNeurons[i];
        network->layers[i].neurons = (struct Neuron*)malloc(numNeurons[i] * sizeof(struct Neuron));

        if (network->layers[i].neurons == NULL) {
            perror("Erreur d'allocation de mémoire pour les neurones de la couche");
            exit(EXIT_FAILURE);
        }

        // Initialisation des poids pour chaque neurone
        int prevLayerNeurons = (i == 0) ? 0 : network->layers[i-1].numNeurons;
        for (int j = 0; j < numNeurons[i]; j++) {
            initializeWeights(&network->layers[i].neurons[j]);
        }
    }

    printf("Initialisation du réseau terminée avec Pooling (%d x %d) -> (%d x %d)\n", width, height, pooledWidth, pooledHeight);
}



//pour le traitement de l'image

//convertir l'image en niveaux de gris
void convertToGrayscale(unsigned char* buffer, int length) {
    // Convertir l'image en niveaux de gris
    for (int i = 0; i < length; i += 4) {
        unsigned char y = buffer[i];  // Composante Y (luminance)
        buffer[i] = y;  // Luminance
        buffer[i + 2] = y;  // Luminance du deuxième pixel
    }
}

// Rognage de l'image
void resizeImage(unsigned char* buffer, int width, int height) {
    int size = width < height ? width : height;
    int offset = (width - size) / 2;

    // Créer un nouveau tampon pour l'image redimensionnée
    unsigned char* newBuffer = malloc(size * size);
    if (!newBuffer) {
        perror("Erreur d'allocation mémoire pour le rognage");
        return;
    }

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int src = (y + offset) * width + x + offset;
            int dst = y * size + x;
            newBuffer[dst] = buffer[src];
        }
    }

    // Copier les données rognées dans le tampon original
    memcpy(buffer, newBuffer, size * size);

    // Libérer la mémoire allouée pour le nouveau tampon
    free(newBuffer);
}

//prétraitement de l'image
void preprocessImage(unsigned char* imageBuffer, int* width, int* height, double* output) {
    int largeur = *width;
    int hauteur = *height;
    // Convertir l'image en niveaux de gris
    convertToGrayscale(imageBuffer, largeur * hauteur * 2);
    //resizeImage(buffer, width, height);

    // Normalisation
    for (int i = 0; i < largeur * hauteur; i++) {
        output[i] = imageBuffer[i] / 255.0;
    }
}


//pour cree le buffer
unsigned char* allocateBuffer(int width, int height) {
    // Calculer la taille nécessaire pour stocker une image en noir et blanc
    unsigned long bufferSize = width * height;
    unsigned char* buffer = (unsigned char*)malloc(bufferSize * sizeof(unsigned char));
    if (!buffer) {
        perror("Erreur d'allocation du buffer");
        return NULL;
    }
    return buffer;
}


// Fonction pour charger la première image et définir la taille du buffer et trouve la hauteur et la largeur de l'image
unsigned char* loadFirstImage(const char* filePath, int* width, int* height) {
    FILE* file = fopen(filePath, "rb");
    if (!file) {
        perror("Erreur d'ouverture du fichier");
        return NULL;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    int row_stride = cinfo.output_width * cinfo.output_components;

    unsigned long bufferSize = row_stride * cinfo.output_height;
    unsigned char* buffer = (unsigned char*)malloc(bufferSize);
    if (!buffer) {
        perror("Erreur d'allocation du buffer");
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return NULL;
    }

    unsigned char* row_pointer = buffer;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
        row_pointer += row_stride;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);

    return buffer;
}


//fonction pour charger les images
unsigned char* loadImage(const char* filePath, int* width, int* height) {
    // Ouvrir le fichier de l'image
    FILE* file = fopen(filePath, "rb");
    if (!file) {
        perror("Erreur lors de l'ouverture de l'image");
        return NULL;
    }

    // Lire les dimensions de l'image
    // Ce code est un exemple simplifié ; vous devrez utiliser une bibliothèque pour lire les dimensions réelles
    *width = 100; // Remplacer par la largeur réelle
    *height = 100; // Remplacer par la hauteur réelle

    int size = (*width) * (*height);
    unsigned char* buffer = (unsigned char*)malloc(size);
    if (!buffer) {
        perror("Erreur lors de l'allocation du buffer");
        fclose(file);
        return NULL;
    }

    // Lire les données de l'image dans le buffer
    size_t bytesRead = fread(buffer, 1, size, file);
    if (bytesRead != size) {
        perror("Erreur lors de la lecture de l'image");
        free(buffer);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return buffer;
}

// Fonction fictive pour charger les étiquettes
double* loadLabel(const char* datasetPath, int index) {
    // Implémentez cette fonction selon la structure de vos étiquettes
    // Exemple: chargement d'une étiquette fictive
    double* label = (double*)malloc(sizeof(double));
    *label = (index % 2); // Exemple: étiquette binaire fictive
    return label;
}


/////////////////////////////////////////////zone d'entrainement////////////////////////////////////
void loadImageDataset(const char* directoryPath, double** inputs, double** labels, int numImages, int* width, int* height) {

    DIR* dir = opendir(directoryPath);
    if (dir == NULL) {
        perror("Erreur lors de l'ouverture du répertoire");
        return;
    }

    if (*width <= 0 || *height <= 0) {
        fprintf(stderr, "Dimensions invalides : width=%d, height=%d\n", *width, *height);
        closedir(dir);
        return;
    }

    //printf("Chargement des images...\n");

    struct dirent* entry;
    int imageCount = 0;
    //initialiser le buffer
    unsigned char* buffer = allocateBuffer(*width, *height);
    //printf("Succès allocation du buffer pour les images...\n");
    if (!buffer) {
        closedir(dir);
        return;
    }
    //buffer = loadFirstImage("photos_visage/Aaron_Peirsol_0002.jpg", width, height);

    while ((entry = readdir(dir)) != NULL) {
        //on vide le buffer initialisé
        memset(buffer, 0, *width * *height);
        // Construire le chemin complet du fichier
        char* filePath[1024];
        char fullFilePath[1024];
        //strcpy(fullFilePath, filePath);
        snprintf(fullFilePath, sizeof(fullFilePath), "%s/%s", directoryPath, entry->d_name);
        // Afficher le chemin du fichier
        //printf("Traitement du fichier : %s", fullFilePath);
        // Ignorer les répertoires '.' et '..'
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            //printf("Chargement de l'image : %s\n", fullFilePath);
            //charger et prétraiter l'image
            //unsigned char* imgBuffer = loadImage(filePath, width, height);
            preprocessImage(buffer, width, height, inputs[imageCount]);
            //printf("Image process avec succès : %s\n", fullFilePath);
            //preprocessImage(cameraImage, &width, &height, buffer);
            //printf("ça marche ici...\n");

            imageCount++;
            if (imageCount >= numImages) {
                break;
            }
        }
    }

    closedir(dir);
}


void trainNetwork(struct Network* network, double** inputs, double** expectedOutputs, int numInputs, int numOutputs, int numEpochs, double learningRate, double** validationInputs, double** validationOutputs, int numValidationSamples) {
    printf("Début de l'initialisation des poids...\n");

    for (int i = 0; i < network->numLayers; i++) {
        printf("Layer %d: %d neurones\n", i, network->layers[i].numNeurons);

        for (int j = 0; j < network->layers[i].numNeurons; j++) {
            int prevLayerNeurons = (i == 0) ? 0 : network->layers[i-1].numNeurons;
            
            if (prevLayerNeurons > 0) {
                if (network->layers[i-1].neurons == NULL) {
                    fprintf(stderr, "Erreur: les neurones de la couche précédente %d ne sont pas alloués\n", i-1);
                    exit(EXIT_FAILURE);
                }
            }

            //printf("Initialisation du neurone %d de la couche %d avec %d neurones dans la couche précédente...\n", j, i, prevLayerNeurons);
            initializeWeights(&network->layers[i].neurons[j]);
        }
    }
    printf("Initialisation des neurones terminée...\n");


    
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        // Traitement des échantillons par mini-batch
        int batchSize = 32; // Exemple de taille de mini-batch
        for (int start = 0; start < numInputs; start += batchSize) {
            int end = (start + batchSize < numInputs) ? start + batchSize : numInputs;
            for (int i = start; i < end; i++) {
                feedForwardNetwork(network, inputs[i]);
                double erreur = cout(network, expectedOutputs[i]);
                backpropagationNetwork(network, expectedOutputs[i], learningRate);
            }
        }

        // Évaluer la précision après chaque époque
        double accuracy = calculateAccuracy(network, validationInputs, validationOutputs, numValidationSamples, numOutputs);
        printf("Époque %d : Précision = %.2f%%\n", epoch + 1, accuracy);
    }
}


// Trouver l'indice de la valeur maximale dans un tableau
int findMaxIndex(double* array, int length) {
    int maxIndex = 0;
    for (int i = 1; i < length; i++) {
        if (array[i] > array[maxIndex]) {
            maxIndex = i;
        }
    }
    return maxIndex;
}

int entrainement_dataset(struct Network network) {
    int width, height;
    const char* firstImagePath = "photos_visage/Aaron_Peirsol_0002.jpg"; // Chemin vers la première image

    // Charger la première image pour obtenir les dimensions
    unsigned char* firstImageBuffer = loadFirstImage(firstImagePath, &width, &height);
    if (!firstImageBuffer) {
        return 1;
    }
    printf("Dimensions de la première image : %d x %d\n", width, height);

    // Application du pooling
    width /= 2;
    height /= 2;
    printf("Dimensions après pooling : %d x %d\n", width, height);

    // Allouer un buffer pour toutes les images du dataset
    printf("Allocation du buffer pour les images...\n");
    unsigned char* buffer = allocateBuffer(width, height);
    //unsigned char* buffer = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    printf("Succès allocation du buffer pour les images...\n");

    if (!buffer) {
        free(firstImageBuffer);
        return 1;
    }

    //Chargement et préparation de la base de données d'images
    int numImages = 13233; // Nombre d'images dans la base de données
    int numClasses = 5749; // Nombre de classes de visages
    double** inputs = (double**)malloc(numImages * sizeof(double*));
    double** labels = (double**)malloc(numImages * sizeof(double*));

    printf("Allocation des images et des étiquettes...\n");
    for (int i = 0; i < numImages; i++) {
        inputs[i] = (double*)malloc(width * height * sizeof(double));//Taille mise à jour
        labels[i] = (double*)malloc(numClasses * sizeof(double));//numClasses est le nombre de classes de visages
    }
    printf("Succès allocation des images et des étiquettes...\n");

    printf("Chargement des images et des étiquettes...\n");
    loadImageDataset("photos_visage", inputs, labels, numImages, &width, &height);
    printf("Succès chargement des images et des étiquettes...\n");

    // Configurer et entraîner le réseau
    int numOutputs = numClasses; // Le nombre de sorties du réseau
    int numEpochs = 10; // Exemple de nombre d'époques
    double learningRate = 0.01; // Exemple de taux d'apprentissage
    trainNetwork(&network, inputs, labels, numImages, numOutputs, numEpochs, learningRate, inputs, labels, numImages);
    printf("Entraînement du réseau terminé...\n");

    // Capture d'images en temps réel pour la reconnaissance 
    /*
    while (1) {
        // Capture et traitement de l'image de la caméra
        // Remplacez ce bloc par une capture réelle si possible
        unsigned char* cameraImage = loadFirstImage("Aaron_Peirsol_0002.jpg", &width, &height); // Exemple d'image
        if (cameraImage) {
            preprocessImage(cameraImage, &width, &height, buffer);
            
            double* prediction = (double*)malloc(numClasses * sizeof(double));
            predict(&network, buffer, prediction);

            // Afficher la prédiction
            printf("Prédiction : %d\n", findMaxIndex(prediction, numClasses));

            free(cameraImage);
            free(prediction);
        }
    }
    */

    // Libérer les ressources
    free(firstImageBuffer);
    free(buffer);
    for (int i = 0; i < numImages; i++) {
        free(inputs[i]);
        free(labels[i]);
    }
    free(inputs);
    free(labels);

    return 0;
}


/////////////////////////////////////////////fin zone d'entrainement////////////////////////////////////

////////////////////////////////////////////zone de test////////////////////////////////////////////


void testInitializeWeights() {
    int numWeights = 5;
    struct Neuron neuron;
    neuron.weights = (double*)malloc(numWeights * sizeof(double));

    initializeWeights(&neuron);

    printf("Test Initialize Weights:\n");
    for (int i = 0; i < numWeights; i++) {
        printf("Weight %d: %f\n", i, neuron.weights);
    }

    printf("Bias: %f\n", neuron.bias);
    free(neuron.weights);
}

void testFeedForward() {
    int numInputs = 3;
    double inputs[] = {1.0, 0.5, -1.5};
    struct Neuron neuron;
    neuron.weights = (double*)malloc(numInputs * sizeof(double));
    
    neuron.weights[0] = 0.5;
    neuron.weights[1] = -1.0;
    neuron.weights[2] = 0.25;
    neuron.bias = 0.1;

    feedForward(&neuron, numInputs, inputs);

    printf("Test Feed Forward:\n");
    printf("Output: %f\n", neuron.output);
    
    free(neuron.weights);
}

void testBackpropagation() {
    int numInputs = 3;
    double inputs[] = {1.0, 0.5, -1.5};
    struct Neuron neuron;
    neuron.weights = (double*)malloc(numInputs * sizeof(double));
    
    neuron.weights[0] = 0.5;
    neuron.weights[1] = -1.0;
    neuron.weights[2] = 0.25;
    neuron.bias = 0.1;
    neuron.output = 0.6; // Supposez que cette sortie a été obtenue après feedforward
    neuron.error = 0.4;  // Supposez une erreur de 0.4
    
    double learningRate = 0.01;
    
    backpropagation(&neuron, numInputs, inputs, learningRate);
    
    printf("Test Backpropagation:\n");
    for (int i = 0; i < numInputs; i++) {
        printf("Updated Weight %d: %f\n", i, neuron.weights[i]);
    }
    printf("Updated Bias: %f\n", neuron.bias);
    
    free(neuron.weights);
}


int test_au_cas_ou(void) {
    testInitializeWeights();
    testFeedForward();
    testBackpropagation();
    return 0;
}

/////////////////////////////////////////////fin zone de test////////////////////////////////////////

// Fonction pour afficher les poids d'un neurone
void printNeuronWeights(struct Network* network, int layerIndex, int neuronIndex) {
    // Assurez-vous que la couche et le neurone existent
    if (layerIndex < network->numLayers && neuronIndex < network->layers[layerIndex].numNeurons) {
        struct Neuron* neuron = &network->layers[layerIndex].neurons[neuronIndex];

        // Assurez-vous que les poids sont alloués
        if (neuron->weights != NULL) {
            int numWeights = (layerIndex == 0) ? 0 : network->layers[layerIndex-1].numNeurons;
            printf("Weights of Neuron %d in Layer %d:\n", neuronIndex, layerIndex);
            printf("Nombre de poids: %d\n", numWeights);  // Afficher le nombre de poids

            // Vérifiez si numWeights est supérieur à 0
            if (numWeights > 0) {
                for (int i = 0; i < numWeights; i++) {
                    printf("Weight %d: %f\n", i, neuron->weights[i]);
                }
            } else {
                printf("Le neurone n'a pas de poids (numWeights = %d).\n", numWeights);
            }
        } else {
            printf("Neuron %d in Layer %d has no allocated weights.\n", neuronIndex, layerIndex);
        }
    } else {
        printf("Layer %d or Neuron %d does not exist.\n", layerIndex, neuronIndex);
    }
}

void printNetworkWeights(struct Network* network) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < network->layers[i].numNeurons; j++) {
            struct Neuron* neuron = &network->layers[i].neurons[j];
            int numWeights = (i == 0) ? 0 : network->layers[i-1].numNeurons;

            printf("Layer %d, Neuron %d: ", i, j);
            if (neuron->weights != NULL) {
                printf("Weights: ");
                printf("%f ", neuron->weights[0]);
                printf("%f ", neuron->weights[1]);
                printf("%f ", neuron->weights[7]);
                printf("\n");
            } else {
                printf("No weights allocated\n");
            }
        }
    }
}


//5749 personnes
//13233 images
int main(void) {
    struct Network mon_reseau;
    int width, height;
    unsigned char* buf = loadFirstImage("photos_visage/Aaron_Peirsol_0002.jpg", &width, &height);
    initializeNetwork(&mon_reseau, height, width, 5749);
    
    /* test pooling (succes)
    // Pooling de l'image
    int pooledWidth = (width - 2) / 2 + 1;
    int pooledHeight = (height - 2) / 2 + 1;
    // Convert buf to double*
    double* bufDouble = (double*)malloc(width * height * sizeof(double));
    for (int i = 0; i < width * height; i++) {
        bufDouble[i] = (double)buf[i];
    }
    double* pooledImage = (double*)malloc(pooledWidth * pooledHeight * sizeof(double));
    
    maxPooling(bufDouble, pooledImage, width, height, 2, 2);

    // Libération de la mémoire
    free(buf);
    free(pooledImage);
    for (int i = 0; i < mon_reseau.numLayers; i++) {
        for (int j = 0; j < mon_reseau.layers[i].numNeurons; j++) {
            printf("Layer %d, Neuron %d, Weights:", i, j);

            if (mon_reseau.layers[i].neurons[j].weights != NULL) {
                for (int k = 0; k < (i == 0 ? 0 : mon_reseau.layers[i-1].numNeurons); k++) {
                    printf(" %f", mon_reseau.layers[i].neurons[j].weights[k]);
                }
            } else {
                printf(" (null)");
            }

            printf("\n");
        }
    }
    */
    int layerIndex = 0; // Index de la couche que vous souhaitez vérifier
    int neuronIndex = 0; // Index du neurone que vous souhaitez vérifier

     // Afficher les poids du neurone spécifié
    //printNeuronWeights(&mon_reseau, 0, 0); // Exemple pour Layer 0, Neuron 0

    // Afficher les poids de tous les neurones
    printNetworkWeights(&mon_reseau);


    //entrainement_dataset(mon_reseau);
    //test_au_cas_ou();
    //main_camera();
    return 0;
}

int main_camera(void) {
    const char* device_path = "/dev/video0"; // Chemin vers le périphérique vidéo
    int fd = open(device_path, O_RDWR);
    if (fd == -1) {
        perror("Erreur lors de l'ouverture du périphérique vidéo");
        return 1;
    }

    // Vérification des capacités de l'appareil
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        perror("Erreur lors de l'interrogation des capacités de l'appareil");
        close(fd);
        return 1;
    }

    // Configuration du format d'image
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 640;   // Largeur
    fmt.fmt.pix.height = 480;  // Hauteur
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;  // Format d'image (YUYV)
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("Erreur lors de la configuration du format d'image");
        close(fd);
        return 1;
    }

    // Configuration de la fréquence d'images
    struct v4l2_streamparm streamparm;
    memset(&streamparm, 0, sizeof(streamparm));
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    streamparm.parm.capture.timeperframe.numerator = 1;
    streamparm.parm.capture.timeperframe.denominator = 10;  // 30 FPS

    if (ioctl(fd, VIDIOC_S_PARM, &streamparm) == -1) {
        perror("Erreur lors de la configuration de la fréquence d'images");
        close(fd);
        return 1;
    }

    // Demander des tampons d'image
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 1;  // Nombre de tampons
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        perror("Erreur lors de la demande de tampons d'image");
        close(fd);
        return 1;
    }

    // Mapping de la mémoire
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;  // L'indice du tampon à mapper

    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
        perror("Erreur lors de la requête de tampon");
        close(fd);
        return 1;
    }

    void* buffer = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (buffer == MAP_FAILED) {
        perror("Erreur lors du mapping de la mémoire");
        close(fd);
        return 1;
    }

    // Activer le streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("Erreur lors de l'activation du streaming");
        munmap(buffer, buf.length);
        close(fd);
        return 1;
    }

    // Capture d'images en boucle
    while (1) {
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            perror("Erreur lors de la mise en file du tampon");
            munmap(buffer, buf.length);
            close(fd);
            return 1;
        }

        if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            perror("Erreur lors de la récupération du tampon");
            munmap(buffer, buf.length);
            close(fd);
            return 1;
        }

        printf("Image capturée, taille : %d octets\n", buf.bytesused);

        // Traitement de l'image capturée
        convertToGrayscale((unsigned char*)buffer, buf.length);
        //resizeImage((unsigned char*)buffer, fmt.fmt.pix.width, fmt.fmt.pix.height);

        // Ici, vous pouvez ajouter du code pour sauvegarder l'image ou la traiter davantage
    }

    // Désactivation du streaming
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
        perror("Erreur lors de la désactivation du streaming");
    }

    // Libérer la mémoire et fermer le périphérique
    munmap(buffer, buf.length);
    close(fd);

    return 0;
}
