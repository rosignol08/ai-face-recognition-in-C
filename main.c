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

struct Neuron{
    double *weights;
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


//structure du neurone
void initializeWeights(struct Neuron* neuron, int n) {
    srand(time(NULL)); // Seed
    for (int i = 0; i < n; i++) {
        neuron->weights[i] = (double)rand() / RAND_MAX * (WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN;
    }
    neuron->bias = (double)rand() / RAND_MAX * (WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN;
}

//fonction d'activation
double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

//pour calculer les sorties (utilisée dans la fonction feedForwardNetwork)
void feedForward(struct Neuron* neuron, int n, double* inputs){
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += neuron->weights[i] * inputs[i];
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
        neuron->weights[i] += learningRate * gradient * inputs[i];
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
                errorSum += nextLayer->neurons[k].weights[j] * nextLayer->neurons[k].error;
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



//pour le traitement de l'image

//convertir l'image en niveaux de gris
void convertToGrayscale(unsigned char* buffer, int length) {
    // Convertir l'image en niveaux de gris
    for (int i = 0; i < length; i += 2) {
        unsigned char y = buffer[i];  // Composante Y (luminance)
        buffer[i] = y;  // Rouge
        buffer[i + 1] = y;  // Vert
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
void preprocessImage(unsigned char* buffer, int width, int height, double* output) {
    convertToGrayscale(buffer, width * height * 2);
    //resizeImage(buffer, width, height);

    // Normalisation
    for (int i = 0; i < width * height; i++) {
        output[i] = buffer[i] / 255.0;
    }
}


//pour cree le buffer
unsigned char* allocateBuffer(int width, int height) {
    return (unsigned char*)malloc(width * height * sizeof(unsigned char));
}

// Fonction pour charger la première image et définir la taille du buffer et trouve la hauteur et la largeur de l'image
unsigned char* loadFirstImage(const char* filePath, int* width, int* height, int* channels) {
    FILE* file = fopen(filePath, "rb");
    if (!file) {
        perror("Erreur d'ouverture du fichier");
        return NULL;
    }

    // Initialiser la bibliothèque libjpeg
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    // Lire les informations du fichier JPEG
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);

    // Extraire les dimensions
    *width = cinfo.image_width;
    *height = cinfo.image_height;
    *channels = cinfo.num_components;

    // Démarrer la décompression
    jpeg_start_decompress(&cinfo);

    // Allouer le buffer pour l'image
    unsigned long bufferSize = *width * *height * *channels;
    unsigned char* buffer = (unsigned char*)malloc(bufferSize);
    if (!buffer) {
        perror("Erreur d'allocation du buffer");
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return NULL;
    }

    // Lire les données de l'image dans le buffer
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* rowPointer = buffer + (cinfo.output_scanline) * (*width) * (*channels);
        jpeg_read_scanlines(&cinfo, &rowPointer, 1);
    }

    // Nettoyer et fermer
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
void loadImageDataset(const char* directoryPath, double** inputs, double** labels, int numImages) {
    DIR* dir = opendir(directoryPath);
    if (dir == NULL) {
        perror("Erreur lors de l'ouverture du répertoire");
        return;
    }

    struct dirent* entry;
    int imageCount = 0;

    while ((entry = readdir(dir)) != NULL) {
        // Construire le chemin complet du fichier
        char filePath[1024];
        snprintf(filePath, sizeof(filePath), "%s/%s", directoryPath, entry->d_name);

        // Ignorer les répertoires '.' et '..'
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            processImage(filePath);
            imageCount++;
            if (imageCount >= numImages) {
                break;
            }
        }
    }

    closedir(dir);
}


// Fonction fictive pour obtenir l'index de la classe à partir du nom du fichier
int getClassIndexFromFilename(const char* filename, int numClasses) {

    return 0; // Retourne l'index de classe fictif
}




void trainNetwork(struct Network* network, double** inputs, double** expectedOutputs, int numInputs, int numOutputs, int numEpochs, double learningRate, double** validationInputs, double** validationOutputs, int numValidationSamples) {
    double erreur = 0;
    for (int i = 0; i < network->numLayers; i++) {
        for (int j = 0; j < network->layers[i].numNeurons; j++) {
            initializeWeights(&network->layers[i].neurons[j], i == 0 ? 0 : network->layers[i-1].numNeurons);
        }
    }
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        for (int i = 0; i < numInputs; i++) {
            feedForwardNetwork(network, inputs[i]);
            erreur = cout(network, expectedOutputs[i]);
            backpropagationNetwork(network, expectedOutputs[i], learningRate);
        }
        // Évaluer la précision après chaque époque
        double accuracy = calculateAccuracy(network, validationInputs, validationOutputs, numValidationSamples, numOutputs);
        printf("Époque %d : Précision = %.2f%%\n", epoch + 1, accuracy);
    }
}
/////////////////////////////////////////////fin zone d'entrainement////////////////////////////////////

////////////////////////////////////////////zone de test////////////////////////////////////////////


void testInitializeWeights() {
    int numWeights = 5;
    struct Neuron neuron;
    neuron.weights = (double*)malloc(numWeights * sizeof(double));

    initializeWeights(&neuron, numWeights);

    printf("Test Initialize Weights:\n");
    for (int i = 0; i < numWeights; i++) {
        printf("Weight %d: %f\n", i, neuron.weights[i]);
        if (neuron.weights[i] < WEIGHT_MIN || neuron.weights[i] > WEIGHT_MAX) {
            printf("Error: Weight out of bounds!\n");
        }
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

//5749 personnes
//13233 images
int entrainement_dataset(void) {
    //le buffer
    unsigned char* buffer = allocateBuffer(640, 480);
    if (!buffer) {
        return 1;
    }
    int width, height, channels;
    const char* firstImagePath = "first_image.jpg"; // Chemin vers la première image
    
    // Charger et préparer la base de données d'images
    int numImages = 13233; // Nombre d'images dans la base de données*
    int numClasses = 5749; // Nombre de classes de visages
    double** inputs = (double**)malloc(numImages * sizeof(double*));
    double** labels = (double**)malloc(numImages * sizeof(double*));
    
    for (int i = 0; i < numImages; i++) {
        inputs[i] = (double*)malloc(64 * 64 * sizeof(double));
        labels[i] = (double*)malloc(numClasses * sizeof(double)); // numClasses est le nombre de classes de visages
    }

    loadImageDataset("photos_visage", inputs, labels, numImages);

    // Configurer et entraîner le réseau
    struct Network network;
    // Initialiser le réseau avec vos couches, par exemple
    // trainNetwork(&network, inputs, labels, numImages, numOutputs, numEpochs, learningRate);

    // Capture d'images en temps réel pour la reconnaissance faciale
    while (1) {
        // Capture et traitement de l'image de la caméra
        // Préparez l'image et passez-la à travers le réseau pour prédire

        double* cameraInput = (double*)malloc(64 * 64 * sizeof(double));
        preprocessImage(buffer, 640, 480, cameraInput);

        double* prediction = (double*)malloc(numClasses * sizeof(double));
        predict(&network, cameraInput, prediction);

        // Afficher la prédiction
        printf("Prédiction : %d\n", findMaxIndex(prediction, numClasses)); // `findMaxIndex` à implémenter pour trouver la classe prédite
    }
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
