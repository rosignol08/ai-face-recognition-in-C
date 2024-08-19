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
#include <SDL2/SDL.h>  // Inclure la bibliothèque SDL2

#define WEIGHT_MAX 10
#define WEIGHT_MIN -10

#include <dirent.h> // Pour opendir, readdir, closedir
#include <sys/stat.h> // Pour S_ISREG et pour DT_REG

#include <errno.h> // Pour errno

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
void initializeWeights(struct Neuron* neuron);
double sigmoid(double x);
void feedForward(struct Neuron* neuron, int n, double* inputs);
void feedForwardNetwork(struct Network* network, double* inputs);
void verifier_integrite_network(struct Network reseau);
void backpropagation(struct Neuron* neuron, int n, double* inputs, double learningRate);
//double cout(struct Network* network, double* expectedOutputs);
void backpropagationNetwork(struct Network* network, double* expectedOutputs, double learningRate);
double calculateAccuracy2(struct Network* network, double** validationInputs, double* validationOutputs, int numValidationSamples);
double calculPrecision(struct Network* reseau, int numImages, double* labels, double** inputs, double seuil);
void maxPooling(double* input, double* output, int inputWidth, int inputHeight, int poolSize, int stride);
void initializeNetwork(struct Network* network, int height, int width, int numOutputs);
void predict(struct Network* network, double* input, double* output);
void loadImageDataset(const char* directoryPath, int numImages, double* labels);
void verifyPredictions(struct Network* network, double** inputs, double* labels, int numImages);
int entrainement_dataset(struct Network network);


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
    //printf("Poids: %f\n", neuron->weights);
    neuron->bias = 0.0; // Valeur de 0 pour le biais
}



//fonction d'activation
double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

//pour calculer les sorties (utilisée dans la fonction feedForwardNetwork)
void feedForward(struct Neuron* neuron, int n, double* inputs) {
    double sum = neuron->bias;  // Commencer par le biais
    for (int i = 0; i < n; i++) {
        sum += neuron->weights * inputs[i];  // Ajouter le produit des poids et des entrées
    }
    neuron->output = sigmoid(sum);  // Appliquer la fonction d'activation
}

//fonction pour vérifier l'intégrité du réseau
void verifier_integrite_network(struct Network reseau) {
    //printf("Vérification de l'intégrité du réseau...\n");
    //printf("Nombre de couches: %d\n", reseau.numLayers);

    int erreur_detectee = 0;

    for (int i = 0; i < reseau.numLayers; i++) {
        //printf("Couche %d: %d neurones\n", i, reseau.layers[i].numNeurons);

        for (int j = 0; j < reseau.layers[i].numNeurons; j++) {
            struct Neuron* neurone = &reseau.layers[i].neurons[j];

            // Vérification de l'allocation des poids
            if (neurone->weights == NAN) {
                printf("Erreur: Neurone %d dans la couche %d n'a pas de poids alloués.\n", j, i);
                erreur_detectee++;
                continue;
            }

            // Vérification du nombre de poids (pour les couches non initiales)
            if (i > 0 && neurone->weights != NAN ){
            //int expectedWeights = reseau.layers[i-1].numNeurons;
                if (isnan(neurone->weights) || isinf(neurone->weights)) {
                    printf("Erreur: Neurone %d dans la couche %d a un poids invalide (NaN ou inf).\n", j, i);
                    erreur_detectee++;
                }
            }

            // Vérification des valeurs de biais
            if (isnan(neurone->bias) || isinf(neurone->bias)) {
                printf("Erreur: Neurone %d dans la couche %d a un biais invalide (NaN ou inf).\n", j, i);
                erreur_detectee++;
            }

            // Affichage des informations sur le neurone
            //printf("Neurone %d: biais = %f\n", j, neurone->bias);
        }
    }

    if (erreur_detectee == 0) {
        printf("Vérification terminée. Aucune erreur détectée.\n");
    } else {
        printf("Vérification terminée avec %d erreurs détectées.\n", erreur_detectee);
    }
}

void feedForwardNetwork(struct Network* network, double* inputs) {
    double* bufferA = malloc(network->layers[0].numNeurons * sizeof(double));
    double* bufferB = malloc(network->layers[0].numNeurons * sizeof(double));
    double* currentInputs = inputs;
    double* currentOutputs = bufferA;

    for (int i = 0; i < network->numLayers; i++) {
        struct Layer* layer = &network->layers[i];

        if (i > 0 && layer->numNeurons != network->layers[i-1].numNeurons) {
            currentOutputs = (currentOutputs == bufferA) ? bufferB : bufferA;
        }

        for (int j = 0; j < layer->numNeurons; j++) {
            feedForward(&layer->neurons[j], (i == 0) ? 0 : network->layers[i-1].numNeurons, currentInputs);
            currentOutputs[j] = layer->neurons[j].output;
        }

        currentInputs = currentOutputs;
    }

    free(bufferA);
    free(bufferB);
}




//ajuster les poids en fonction de l'erreur (utilisée dans la fonction backpropagationNetwork)
void backpropagation(struct Neuron* neuron, int n, double* inputs, double learningRate){
    //printf("Backpropagation: commence\n");
    double gradient = neuron->output * (1 - neuron->output) * neuron->error;
    for (int i = 0; i < n; i++) {
        neuron->weights += learningRate * gradient * inputs[i];
    }
    neuron->bias += learningRate * gradient;
    //printf("Backpropagation: fin\n");
    return;
}

double cout(struct Network* network, double* expectedOutputs) {
    printf("Calcul de l'erreur: commence\n");
    double error = 0.0;
    struct Layer* outputLayer = &network->layers[network->numLayers - 1];
    printf("outputLayer->numNeurons: %d\n", outputLayer->numNeurons);

    // Assurez-vous que les tailles correspondent
    if (outputLayer->numNeurons != sizeof(expectedOutputs) / sizeof(expectedOutputs[0])) {
        fprintf(stderr, "Erreur: la taille des sorties attendues ne correspond pas à la taille de la couche de sortie\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < outputLayer->numNeurons; i++) {
        double predicted = outputLayer->neurons[i].output;  // Accéder à la sortie du neurone
        double expected = expectedOutputs[i];  // Accéder à la sortie attendue
        printf("predicted: %f, expected: %f\n", predicted, expected);
        error += pow(expected - predicted, 2);  // Erreur quadratique
    }

    // Calculer la moyenne de l'erreur
    error /= outputLayer->numNeurons;
    
    printf("Calcul de l'erreur: fin\n");
    printf("Erreur = %f\n", error);
    return error;
}



//ajuster les poids en fonction de l'erreur
void backpropagationNetwork(struct Network* network, double* expectedOutputs, double learningRate) {
    printf("Backpropagation Network: commence\n");
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
    printf("Backpropagation Network: fin\n");
}

double calculPrecision(struct Network* reseau, int numImages, double* labels, double** inputs, double seuil) {
    double resultat = 0.0;
    for (int i = 0; i < numImages; i++) {
        double prediction;
        predict(reseau, inputs[i], &prediction);

        // Vérifiez si la prédiction est suffisamment proche du label attendu
        if (fabs(prediction - labels[i]) <= seuil) {
            resultat++;
        }
    }
    double accuracy = (resultat / numImages) * 100.0;
    printf("Précision: %.2f%%\n", accuracy);
    return accuracy;
}




//fonction pour calculer la précision
double calculateAccuracy2(struct Network* network, double** validationInputs, double* validationOutputs, int numValidationSamples) {
    printf("Calcul de la précision...\n");
    int correctPredictions = 0;
    double predictedOutput;

    for (int i = 0; i < numValidationSamples; i++) {
        predict(network, validationInputs[i], &predictedOutput);  // Prédiction du réseau pour une seule sortie

        // Puisque l'étiquette est 0.0 ou 1.0, on arrondit la prédiction à la classe la plus proche
        int predictedClass = (predictedOutput >= 0.5) ? 1 : 0;
        int actualClass = (int)validationOutputs[i];

        if (predictedClass == actualClass) {
            correctPredictions++;
        }
    }

    double accuracy = (double)correctPredictions / numValidationSamples * 100.0;
    printf("Précision: %.2f%%\n", accuracy);
    return accuracy;
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
    
    // Dimensions après pooling
    int poolSize = 2;
    int stride = 2;
    int pooledWidth = (width - poolSize) / stride + 1;
    int pooledHeight = (height - poolSize) / stride + 1;
    int numInputs = pooledWidth * pooledHeight;
    
    // Exemple de configuration du réseau
    int numNeurons[] = {numInputs, 512, 256, 128, 64, 16, numOutputs};
    network->numLayers = sizeof(numNeurons) / sizeof(numNeurons[0]); // Nombre de couches
    network->layers = (struct Layer*)malloc(network->numLayers * sizeof(struct Layer));
    printf("il y a %d couches\n", network->numLayers);

    if (network->layers == NULL) {
        perror("Erreur d'allocation de mémoire pour les couches du réseau");
        exit(EXIT_FAILURE);
    }


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
        //int prevLayerNeurons = (i == 0) ? 0 : network->layers[i-1].numNeurons;
        for (int j = 0; j < numNeurons[i]; j++) {
            initializeWeights(&network->layers[i].neurons[j]);
        }
    }

    printf("Initialisation du réseau terminée avec Pooling (%d x %d) -> (%d x %d)\n", width, height, pooledWidth, pooledHeight);
    printf("Initialisation du réseau terminée avec %d neurones d'entrée, %d neurones cachés et %d neurones de sortie\n", numInputs, numNeurons[1], numOutputs);
}



//pour le traitement de l'image
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
    if (bytesRead != (size_t)size) {
        perror("Erreur lors de la lecture de l'image");
        free(buffer);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return buffer;
}


/////////////////////////////////////////////zone d'entrainement////////////////////////////////////
void trainNetwork(struct Network* network, double** inputs, double* expectedOutputs, int numInputs, int numOutputs, int numEpochs, double learningRate, double** validationInputs, double** validationOutputs, int numValidationSamples) {
    printf("Début de l'initialisation des poids...\n");
    for (int i = 0; i < numInputs; i++) {
        printf("expectedOutputs[%d]: %f\n", i,expectedOutputs[i]);
    }
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
        printf("Époque %d : Entraînement du réseau...\n", epoch + 1);
        // Traitement des échantillons par mini-batch
        int batchSize = 32; // Exemple de taille de mini-batch
        for (int start = 0; start < numInputs; start += batchSize) {
            int end = (start + batchSize < numInputs) ? start + batchSize : numInputs;
            for (int i = start; i < end; i++) {
                printf("Traitement de l'échantillon %d...\n", i);
                feedForwardNetwork(network, inputs[i]);
                printf("expectedOutputs[i]: %f\n", expectedOutputs[i]);
                double erreur = cout(network, &expectedOutputs[i]);
                //printf("Erreur = %.2f\n", erreur);
                backpropagationNetwork(network, &expectedOutputs[i], learningRate);
            }
        }
        //double reponse_du_reseau = &network->layers[network->numLayers-1].neurons->output;
        // Évaluer la précision après chaque époque
        //calculateAccuracy(network, validationInputs, validationOutputs, numValidationSamples, numOutputs);
        double seuil = 0.2;
        double accuracy = calculPrecision(network, numValidationSamples, expectedOutputs, validationOutputs, seuil);
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

/////////////////////////////zone d'entrainement2////////////////////////////////////
// Fonction pour prédire les sorties
void predict(struct Network* network, double* input, double* output) {
    printf("Démarrage de predict...\n");
    feedForwardNetwork(network, input);
    // Copier les sorties du dernier layer
    struct Layer* outputLayer = &network->layers[network->numLayers - 1];
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        output[i] = outputLayer->neurons[i].output;
    }
    printf("Fin de predict...\n");
}

// Fonction pour charger le dataset d'images
void loadImageDataset(const char* directoryPath, int numImages, double* labels) {
    DIR* dir = opendir(directoryPath);
    if (dir == NULL) {
        perror("Erreur lors de l'ouverture du répertoire");
        return;
    }

    struct dirent* entry;
    int imageCount = 0;

    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".jpg") != NULL) {
            printf("Nom du fichier : %s\n", entry->d_name);

            // Déterminer l'étiquette en fonction du nom de fichier
            if (entry->d_name[0] == 'v') {
                labels[imageCount] = 1.0; // Il y a un visage
                printf("Visage détecté\n");
            } else if (entry->d_name[0] == 'p') {
                labels[imageCount] = 0.0; // Pas de visage
                printf("Pas de visage\n");
            } else {
                fprintf(stderr, "Nom de fichier invalide : %s\n", entry->d_name);
                labels[imageCount] = 0.0; // Par défaut, pas de visage
                printf("Nom de fichier non reconnu\n");
            }

            printf("Étiquette : %f, et image numero: %d \n", labels[imageCount], imageCount);

            imageCount++;
            if (imageCount >= numImages) {
                break;
            }
        }
    }
    closedir(dir);
}

// Fonction pour vérifier les prédictions
void verifyPredictions(struct Network* network, double** inputs, double* labels, int numImages) {
    printf("Vérification des prédictions...\n");
    double* prediction = (double*)malloc(sizeof(double));
    if (prediction == NULL) {
        perror("Erreur d'allocation pour prediction");
        exit(EXIT_FAILURE);
    }

    int correctPredictions = 0;
    for (int i = 0; i < numImages; i++) {
        predict(network, inputs[i], prediction);
        int predictedLabel = (prediction[0] >= 0.5) ? 1 : 0;
        int actualLabel = (labels[i] >= 0.5) ? 1 : 0;

        if (predictedLabel == actualLabel) {
            correctPredictions++;
        }
    }

    double accuracy = (double)correctPredictions / numImages * 100.0;
    printf("Précision des vérifications : %.2f%%\n", accuracy);

    free(prediction);
}

int entrainement_dataset(struct Network network) {
    //struct Network *copie_network = &network;

    int width, height;
    const char* firstImagePath = "photoss_visage/v0.jpg"; // Chemin vers la première image

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
    if (!buffer) {
        free(firstImageBuffer);
        return 1;
    }

    // Chargement et préparation de la base de données d'images
    int numImages = 20; // Nombre d'images dans la base de données
    int numClasses = 1; // Nombre de classes de visages (utilisé plus tard)
    double** inputs = (double**)malloc(numImages * sizeof(double*));
    double* labels = (double*)malloc(numImages * sizeof(double));

    if (inputs == NULL || labels == NULL) {
        fprintf(stderr, "Erreur d'allocation de mémoire pour inputs ou labels\n");
        return 1;
    }

    printf("Allocation des images et des étiquettes...\n");
    for (int i = 0; i < numImages; i++) {
        inputs[i] = (double*)malloc(width * height * sizeof(double)); // Taille mise à jour
        if (inputs[i] == NULL) {
            fprintf(stderr, "Erreur d'allocation de mémoire pour inputs[%d]\n", i);
            return 1;
        }
    }
    printf("Succès allocation des images et des étiquettes...\n");

    // Initialisation des labels à zéro pour éviter les valeurs aléatoires
    for (int i = 0; i < numImages; i++) {
        labels[i] = 0.0;
    }

    printf("Chargement des images et des étiquettes...\n");
    loadImageDataset("photoss_visage", numImages, labels);
    printf("Succès chargement des images et des étiquettes...\n");

    // Configurer et entraîner le réseau
    int numOutputs = numClasses; // Le nombre de sorties du réseau
    int numEpochs = 5; // Exemple de nombre d'époques
    double learningRate = 0.51; // Exemple de taux d'apprentissage
    //for(int i = 0; i < numImages; i++) {
    //    printf("labels[%d]: %f\n", i, labels[i]);
    //}
    trainNetwork(&network, inputs, labels, numImages, numOutputs, numEpochs, learningRate, inputs, &labels, numImages);
    printf("Entraînement du réseau terminé...\n");

    // Vérifier les prédictions
    //verifyPredictions(&network, inputs, labels, numImages);

    // Libérer les ressources
    free(firstImageBuffer);
    free(buffer);
    for (int i = 0; i < numImages; i++) {
        free(inputs[i]);
    }
    free(inputs);
    free(labels);

    return 0;
}
/////////////////////////////////////////////fin zone d'entrainement2////////////////////////////////////



/////////////////////////////////////////////fin zone d'entrainement////////////////////////////////////

////////////////////////////////////////////zone de test////////////////////////////////////////////

void testInitializeWeights() {
    int numWeights = 5;
    struct Neuron neuron;
    initializeWeights(&neuron);

    printf("Test Initialize Weights:\n");
    for (int i = 0; i < numWeights; i++) {
        printf("Weight %d: %f\n", i, neuron.weights);
    }

    printf("Bias: %f\n", neuron.bias);
}

void testFeedForward() {
    int numInputs = 3;
    double inputs[] = {1.0, 0.5, -1.5};
    struct Neuron neuron;
    if (neuron.weights == NAN) {
        neuron.weights = 0.0;//au cas ou ça plante
    }

    neuron.weights = 0.5;
    neuron.bias = 0.1;

    feedForward(&neuron, numInputs, inputs);

    printf("Test Feed Forward:\n");
    printf("Output: %f\n", neuron.output);
}

void testBackpropagation() {
    int numInputs = 3;
    double inputs[] = {1.0, 0.5, -1.5};
    struct Neuron neuron;
    if (neuron.weights == NAN) {
        neuron.weights = 0.0;//au cas ou ça plante
    }
    neuron.weights = 0.5;
    neuron.bias = 0.1;
    neuron.output = 0.6; // Supposez que cette sortie a été obtenue après feedforward
    neuron.error = 0.4;  // Supposez une erreur de 0.4

    double learningRate = 0.01;

    backpropagation(&neuron, numInputs, inputs, learningRate);

    printf("Test Backpropagation:\n");
    for (int i = 0; i < numInputs; i++) {
        printf("Updated Weight %d: %f\n", i, neuron.weights);
    }
    printf("Updated Bias: %f\n", neuron.bias);
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
        if (neuron->weights != NAN) {
            int numWeights = (layerIndex == 0) ? 0 : network->layers[layerIndex-1].numNeurons;
            printf("Weights of Neuron %d in Layer %d:\n", neuronIndex, layerIndex);
            printf("Nombre de poids: %d\n", numWeights);  // Afficher le nombre de poids

            // Vérifiez si numWeights est supérieur à 0
            if (numWeights > 0) {
                for (int i = 0; i < numWeights; i++) {
                    printf("Weight %d: %f\n", i, neuron->weights);
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
            //int numWeights = (i == 0) ? 0 : network->layers[i-1].numNeurons;

            printf("Layer %d, Neuron %d: ", i, j);
            if (neuron->weights != NAN) {
                printf("Weights: ");
                printf("%f ", neuron->weights);
                printf("\n");
            } else {
                printf("No weights allocated\n");
            }
        }
    }
}



void preprocessAndPredict2(struct Network* network, void* buffer, int width, int height, unsigned char* buffer_result) {

    double* input = (double*)malloc(width * height * sizeof(double));
    if (input == NULL) {
        perror("Erreur d'allocation mémoire pour l'entrée");
        return;
    }

    // Convertir l'image capturée en un format double
    unsigned char* imageBuffer = (unsigned char*)buffer;
    int imageSize = width * height;
    double* inputs = (double*)malloc(1 * sizeof(double*));
    //printf("avant: %d\n", buffer_result[0]);
    preprocessImage(buffer_result, &width, &height, input);
    printf("après: %d\n", buffer_result[0]);
    ////for (int i = 0; i < 10; i++) {
    //    printf("input[%d] = %f\n", i, input[i]);
    //}
    printf("avant, network: %f, input: %f \n", network->layers[network->numLayers - 1].neurons[0].weights, input[0]);
    feedForwardNetwork(network, input);
    printf("après, network: %f, input: %f \n", network->layers[network->numLayers - 1].neurons[0].weights, input[0]);

    // Récupérer les sorties du réseau
    printf("Démarrage de predict...\n");
    feedForwardNetwork(network, input);
    // Copier les sorties du dernier layer
    struct Layer* outputLayer = &network->layers[network->numLayers - 1];

    double* output = (double*)malloc(outputLayer->numNeurons * sizeof(double));

    for (int i = 0; i < outputLayer->numNeurons; i++) {
        output[i] = outputLayer->neurons[i].output;
    }
    printf("Fin de predict...\n");
    int numOutputs = network->layers[4].numNeurons;
    double prediction;
    
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        output[i] = outputLayer->neurons[i].output;
    }

    printf("neuron %d, de la couche %d\n", outputLayer->numNeurons, network->numLayers - 1);

    double predictede = outputLayer->neurons[0].output;
    static double previousSum = 0;
    double currentSum = 0;
    for (int i = 0; i < width * height; i++) {
        currentSum += input[i];
    }
    
    if (fabs(currentSum - previousSum) < 1e-5) {
        printf("Alerte : L'image actuelle est très similaire à la précédente\n");
    }
    //printf("precious sum = %f\n", previousSum);
    //printf("current sum = %f\n", currentSum);
    previousSum = currentSum;

    printf("Prédiction : %f\n", output[0]);
    //for (int i = 0; i < numOutputs; i++) {
    //    output[i] = network->layers[network->numLayers - 1].neurons[i].output;
    //}

    // Ajouter des impressions pour déboguer
    //printf("Image traitée :\n");
   // for (int i = 0; i < width * height; i += (width * height) / 10) {
        //printf("input[%d] = %f\n", i, input[i]);
    //}

    //printf("Prédiction :\n");
    //for (int i = 0; i < numOutputs; i++) {
    //    printf("output[%d] = %f\n", i, output[i]);
    //}

    //int detectedFace = (predictede > 0.5) ? 1 : 0;
    
    if (output[0] > 0.5) {
        printf("Visage détecté !\n");
    } else {
        printf("Pas de visage détecté.\n");
    }

    free(input);
    free(output);
}

void preprocessAndPredict(struct Network* network, void* buffer, int width, int height, unsigned char* buffer_result) {
    // Allocation mémoire pour l'entrée du réseau (taille en fonction de l'image d'entrée)
    double* input = (double*)malloc(width * height * sizeof(double));
    if (input == NULL) {
        perror("Erreur d'allocation mémoire pour l'entrée");
        return;
    }

    // Prétraitement de l'image (mise à l'échelle, normalisation, etc.)
    preprocessImage(buffer_result, &width, &height, input);

    // Propagation avant dans le réseau de neurones
    feedForwardNetwork(network, input);

    // Récupérer la sortie du réseau
    struct Layer* outputLayer = &network->layers[network->numLayers - 1];
    double* output = (double*)malloc(outputLayer->numNeurons * sizeof(double));
    if (output == NULL) {
        perror("Erreur d'allocation mémoire pour la sortie");
        free(input);
        return;
    }

    for (int i = 0; i < outputLayer->numNeurons; i++) {
        output[i] = outputLayer->neurons[i].output;
    }

    // Affichage de la prédiction
    printf("Prédiction :\n");
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        printf("Output neuron %d: %f\n", i, output[i]);
    }

    // Vérification simple d'un visage détecté (hypothèse : sortie binaire 0 ou 1)
    if (output[0] > 0.5) {
        printf("Visage détecté !\n");
    } else {
        printf("Pas de visage détecté.\n");
    }

    // Libération de la mémoire
    free(input);
    free(output);
}


void testNetwork(struct Network* network, void* testBuffer, int width, int height) {
    struct Layer* outputLayer = &network->layers[network->numLayers - 1];
    struct  Neuron* outputNeuron = &outputLayer->neurons[0];
    

    // Convertir l'image capturée en un format double pour le réseau
    double* input = (double*)malloc(width * height * sizeof(double));
    if (input == NULL) {
        perror("Erreur d'allocation mémoire pour l'entrée");
        return;
    }

    unsigned char* imageBuffer = (unsigned char*)testBuffer;
    int imageSize = width * height;

    for (int i = 0; i < imageSize; i++) {
        input[i] = imageBuffer[i] / 255.0; // Normalisation
    }

    // Faire passer les données d'entrée à travers le réseau
    feedForwardNetwork(network, input);

    // Récupérer les sorties du réseau
    double* output = (double*)malloc(outputLayer->numNeurons * sizeof(double));
    if (output == NULL) {
        perror("Erreur d'allocation mémoire pour la sortie");
        free(input);
        return;
    }

    for (int i = 0; i < outputLayer->numNeurons; i++) {
        output[i] = outputLayer->neurons[i].output;
    }

    // Afficher les résultats
    printf("Résultats des tests :\n");
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        printf("Output neuron %d: %f\n", i, output[i]);
    }
    double test_sortie = outputNeuron->output;
    // Décider si un visage est détecté
    if (test_sortie > 0.5) {
        printf("Visage détecté !\n");
    } else {
        printf("Pas de visage détecté.\n");
    }
    //printf("couches d'entrée: %d\n", network->layers[0].numNeurons);
    free(input);
    free(output);
}


int main_camera(struct Network* mon_reseau, unsigned char* buffer_result) {
    int width = 640, height = 480; // Valeurs par défaut
    
    

    // Initialiser SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "Erreur SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    // Créer une fenêtre SDL
    SDL_Window* window = SDL_CreateWindow("Camera Feed",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          width,
                                          height,
                                          SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "Erreur SDL_CreateWindow: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // Créer un rendu SDL
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Erreur SDL_CreateRenderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Créer une texture SDL pour afficher l'image
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_YUY2, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!texture) {
        fprintf(stderr, "Erreur SDL_CreateTexture: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Ouverture du périphérique vidéo
    const char* device_path = "/dev/video0"; // Chemin vers le périphérique vidéo
    int fd = open(device_path, O_RDWR);
    if (fd == -1) {
        perror("Erreur lors de l'ouverture du périphérique vidéo");
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Vérification des capacités de l'appareil
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        perror("Erreur lors de l'interrogation des capacités de l'appareil");
        close(fd);
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
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
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Configuration de la fréquence d'images
    struct v4l2_streamparm streamparm;
    memset(&streamparm, 0, sizeof(streamparm));
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    streamparm.parm.capture.timeperframe.numerator = 1;
    streamparm.parm.capture.timeperframe.denominator = 1;  // 30 FPS
    if (ioctl(fd, VIDIOC_S_PARM, &streamparm) == -1) {
        perror("Erreur lors de la configuration de la fréquence d'images");
        close(fd);
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
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
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Mapping de la mémoire
    struct v4l2_buffer videoBuffer;
    memset(&videoBuffer, 0, sizeof(videoBuffer));
    videoBuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    videoBuffer.memory = V4L2_MEMORY_MMAP;
    videoBuffer.index = 0;  // L'indice du tampon à mapper
    if (ioctl(fd, VIDIOC_QUERYBUF, &videoBuffer) == -1) {
        perror("Erreur lors de la requête de tampon");
        close(fd);
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    void* buffer = mmap(NULL, videoBuffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, videoBuffer.m.offset);
    if (buffer == MAP_FAILED) {
        perror("Erreur lors du mapping de la mémoire");
        close(fd);
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Activer le streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("Erreur lors de l'activation du streaming");
        munmap(buffer, videoBuffer.length);
        close(fd);
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Capture d'images en boucle
    SDL_Event e;
    int quit = 0;

    while (!quit) {
        // Capturer une image
        if (ioctl(fd, VIDIOC_QBUF, &videoBuffer) == -1) {
            perror("Erreur lors de la mise en file du tampon");
            break;
        }

        if (ioctl(fd, VIDIOC_DQBUF, &videoBuffer) == -1) {
            perror("Erreur lors de la récupération du tampon");
            break;
        }

        // Afficher l'image avec SDL
        SDL_UpdateTexture(texture, NULL, buffer, width * 2); // YUYV a une largeur de 2 octets par pixel
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        // Gérer les événements SDL (par exemple, fermer la fenêtre)
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = 1;
            }
        }

        // Copier l'image capturée dans un autre buffer pour la prédiction
        memcpy(buffer_result, buffer, width * height);

        // Traitement de l'image capturée pour la reconnaissance de visage
        //preprocessAndPredict(mon_reseau, (unsigned char*)buffer, width, height, buffer_result);
        testNetwork(mon_reseau, (unsigned char*)buffer, width, height);
        // Ajoutez une pause ou une condition pour sortir de la boucle si nécessaire
        SDL_Delay(1); // Pour une fréquence d'images d'environ 30 FPS
    }


    // Désactivation du streaming
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
        perror("Erreur lors de la désactivation du streaming");
    }

    // Libérer la mémoire et fermer le périphérique
    munmap(buffer, videoBuffer.length);
    close(fd);

    // Nettoyer SDL
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
//5749 personnes
//13233 images
int main(void) {
    struct Network mon_reseau;
    int width, height;
    loadFirstImage("photoss_visage/v0.jpg", &width, &height);
    unsigned char* buffer_result = allocateBuffer(width, height);
    initializeNetwork(&mon_reseau, width, height, 1);
    //entrainement_dataset(mon_reseau);
    printf("fin de l'entrainement\n");
    //printf("valeur de poid du neurone 0 de la couche 0: %f\n", mon_reseau.layers[4].neurons[0].weights[0]);
    // Load the first image
    unsigned char* image1 = loadFirstImage("photoss_visage/v0.jpg", &width, &height);
    if (!image1) {
        return 1;
    }

    // Process and predict the first image
    preprocessAndPredict(&mon_reseau, image1, width, height, buffer_result);

    // Load the second image
    unsigned char* image2 = loadFirstImage("photoss_visage/p0.jpg", &width, &height);
    if (!image2) {
        return 1;
    }

    // Process and predict the second image
    preprocessAndPredict(&mon_reseau, image2, width, height, buffer_result);

    // Free the memory
    free(image1);
    free(image2);
    //main_camera(&mon_reseau, buffer_result);
    /*
    struct Network mon_reseau;
    int width, height;
    unsigned char* buf = loadFirstImage("photos_visage/Aaron_Peirsol_0002.jpg", &width, &height);
    if (!buf) {
        return 1;
    }
    //5749
    initializeNetwork(&mon_reseau, height, width, 17);
    
     test pooling (succes)
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
    //int layerIndex = 0; // Index de la couche que vous souhaitez vérifier
    //int neuronIndex = 0; // Index du neurone que vous souhaitez vérifier

     // Afficher les poids du neurone spécifié
    //printNeuronWeights(&mon_reseau, 0, 0); // Exemple pour Layer 0, Neuron 0

    // Afficher les poids de tous les neurones
    //printNetworkWeights(&mon_reseau);


    entrainement_dataset(mon_reseau);
    */

    //test_au_cas_ou();
    //main_camera();
    return 0;
}