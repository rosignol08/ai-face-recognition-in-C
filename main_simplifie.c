#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <SDL2/SDL.h>
#include <errno.h>
#include <dirent.h>

#define INPUT_WIDTH 400
#define INPUT_HEIGHT 400
#define INPUT_NODES (INPUT_WIDTH * INPUT_HEIGHT)  // 2500 pixels
#define HIDDEN_LAYER1_NODES 256
#define HIDDEN_LAYER2_NODES 64
#define OUTPUT_NODES 2
#define LEARNING_RATE 0.1

#define CAM_DEVICE "/dev/video0"
#define WIDTH 1280
#define HEIGHT 720
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720


typedef struct {
    unsigned char *buffer;
    int width;
    int height;
    int channels;
} Image;

typedef struct {
    double **weights1;
    double *bias1;
    double **weights2;
    double *bias2;
    double **output_weights;
    double *output_bias;
} NeuralNetwork;

void initNetwork(NeuralNetwork *nn) {

    srand(time(NULL));
    nn->weights1 = (double **)malloc(INPUT_NODES * sizeof(double *));
    for (int i = 0; i < INPUT_NODES; ++i) {
        nn->weights1[i] = (double *)malloc(HIDDEN_LAYER1_NODES * sizeof(double));
        for (int j = 0; j < HIDDEN_LAYER1_NODES; ++j) {
            nn->weights1[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    nn->bias1 = (double *)malloc(HIDDEN_LAYER1_NODES * sizeof(double));
    for (int j = 0; j < HIDDEN_LAYER1_NODES; ++j) {
        nn->bias1[j] = ((double)rand() / RAND_MAX) * 2 - 1;
    }


    nn->weights2 = (double **)malloc(HIDDEN_LAYER1_NODES * sizeof(double *));
    for (int j = 0; j < HIDDEN_LAYER1_NODES; ++j) {
        nn->weights2[j] = (double *)malloc(HIDDEN_LAYER2_NODES * sizeof(double));
        for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
            //printf("k = %d\n", k);
            nn->weights2[j][k] = ((double)rand() / RAND_MAX) * 2 - 1;
            printf("nn->weights2[%d][%d] = %f\n", j, k, nn->weights2[j][k]);
        }
    }

    printf("Initialisation des poids et biais pour la première couche cachée...\n");
    nn->bias2 = (double *)malloc(HIDDEN_LAYER2_NODES * sizeof(double));
    for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
        nn->bias2[k] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    nn->output_weights = (double **)malloc(HIDDEN_LAYER2_NODES * sizeof(double *));
    for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
        nn->output_weights[k] = (double *)malloc(OUTPUT_NODES * sizeof(double));
        for (int l = 0; l < OUTPUT_NODES; ++l) {
            nn->output_weights[k][l] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    nn->output_bias = (double *)malloc(OUTPUT_NODES * sizeof(double));
    for (int l = 0; l < OUTPUT_NODES; ++l) {
        nn->output_bias[l] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

void freeNetwork(NeuralNetwork *nn) {
    for (int i = 0; i < INPUT_NODES; ++i) {
        free(nn->weights1[i]);
    }
    free(nn->weights1);
    free(nn->bias1);

    for (int j = 0; j < HIDDEN_LAYER1_NODES; ++j) {
        free(nn->weights2[j]);
    }
    free(nn->weights2);
    free(nn->bias2);

    for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
        free(nn->output_weights[k]);
    }
    
    free(nn->output_weights);
    free(nn->output_bias);
}


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void feedForward(NeuralNetwork *nn, double *input, double *hiddenLayer1Output, double *hiddenLayer2Output, double *outputLayerOutput) {
    for (int j = 0; j < HIDDEN_LAYER1_NODES; ++j) {
        hiddenLayer1Output[j] = nn->bias1[j];
        for (int i = 0; i < INPUT_NODES; ++i) {
            hiddenLayer1Output[j] += input[i] * nn->weights1[i][j];
        }
        hiddenLayer1Output[j] = sigmoid(hiddenLayer1Output[j]);
    }

    for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
        hiddenLayer2Output[k] = nn->bias2[k];
        for (int j = 0; j < HIDDEN_LAYER1_NODES; ++j) {
            hiddenLayer2Output[k] += hiddenLayer1Output[j] * nn->weights2[j][k];
        }
        hiddenLayer2Output[k] = sigmoid(hiddenLayer2Output[k]);
    }


    for (int l = 0; l < OUTPUT_NODES; ++l) {
        outputLayerOutput[l] = nn->output_bias[l];
        for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
            outputLayerOutput[l] += hiddenLayer2Output[k] * nn->output_weights[k][l];
        }
        outputLayerOutput[l] = sigmoid(outputLayerOutput[l]);
    }
}

void trainNetwork(NeuralNetwork *nn, double *input, double *targetOutput, int epoch) {
    double hiddenLayer1Output[HIDDEN_LAYER1_NODES];
    double hiddenLayer2Output[HIDDEN_LAYER2_NODES];
    double outputLayerOutput[OUTPUT_NODES];
    double outputLayerError[OUTPUT_NODES];
    double hiddenLayer2Error[HIDDEN_LAYER2_NODES];
    double hiddenLayer1Error[HIDDEN_LAYER1_NODES];

    feedForward(nn, input, hiddenLayer1Output, hiddenLayer2Output, outputLayerOutput);

    // Calcul de l'erreur de sortie
    for (int l = 0; l < OUTPUT_NODES; ++l) {
        double error = targetOutput[l] - outputLayerOutput[l];
        outputLayerError[l] = error * outputLayerOutput[l] * (1 - outputLayerOutput[l]);
    }

    // Calcul de l'erreur pour la deuxième couche cachée
    for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
        double error = 0.0;
        for (int l = 0; l < OUTPUT_NODES; ++l) {
            error += outputLayerError[l] * nn->output_weights[k][l];
        }
        hiddenLayer2Error[k] = error * hiddenLayer2Output[k] * (1 - hiddenLayer2Output[k]);
    }

    // Calcul de l'erreur pour la première couche cachée
    for (int j = 0; j < HIDDEN_LAYER1_NODES; ++j) {
        double error = 0.0;
        for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
            error += hiddenLayer2Error[k] * nn->weights2[j][k];
        }
        hiddenLayer1Error[j] = error * hiddenLayer1Output[j] * (1 - hiddenLayer1Output[j]);
    }

    // Mise à jour des poids et biais pour la couche de sortie
    for (int l = 0; l < OUTPUT_NODES; ++l) {
        nn->output_bias[l] += LEARNING_RATE * outputLayerError[l];
        for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
            nn->output_weights[k][l] += LEARNING_RATE * outputLayerError[l] * hiddenLayer2Output[k];
        }
    }

    // Mise à jour des poids et biais pour la deuxième couche cachée
    for (int k = 0; k < HIDDEN_LAYER2_NODES; ++k) {
        nn->bias2[k] += LEARNING_RATE * hiddenLayer2Error[k];
        for (int j = 0; j < HIDDEN_LAYER1_NODES; ++j) {
            nn->weights2[j][k] += LEARNING_RATE * hiddenLayer2Error[k] * hiddenLayer1Output[j];
        }
    }

    // Mise à jour des poids et biais pour la première couche cachée
    for (int j = 0; j < HIDDEN_LAYER1_NODES; ++j) {
        nn->bias1[j] += LEARNING_RATE * hiddenLayer1Error[j];
        for (int i = 0; i < INPUT_NODES; ++i) {
            nn->weights1[i][j] += LEARNING_RATE * hiddenLayer1Error[j] * input[i];
        }
    }

    // Affichage des statistiques d'entraînement
    double totalError = 0.0;
    for (int l = 0; l < OUTPUT_NODES; ++l) {
        totalError += fabs(targetOutput[l] - outputLayerOutput[l]);
    }

    printf("Epoch %d: Total Error: %f\n", epoch, totalError);
    for (int l = 0; l < OUTPUT_NODES; ++l) {
        printf("Output[%d]: %f\n", l, outputLayerOutput[l]);
    }
}

// Structure pour les tampons
typedef struct {
    unsigned char *start;
    size_t length;
} Buffer;



void xioctl(int fd, int request, void *arg) {
    if (ioctl(fd, request, arg) == -1) {
        perror("ioctl");
        exit(EXIT_FAILURE);
    }
}
// Fonction pour capturer et traiter l'image
void captureAndProcessImage(double *buffer) {
    int fd = open(CAM_DEVICE, O_RDWR);
    if (fd == -1) {
        perror("Ouverture du périphérique de la caméra échouée");
        exit(EXIT_FAILURE);
    }

    // Configuration du format de capture
    struct v4l2_format format = {0};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24; // Format RGB
    format.fmt.pix.width = 1280;
    format.fmt.pix.height = 720;
    xioctl(fd, VIDIOC_S_FMT, &format);

    // Demande de tampons
    struct v4l2_requestbuffers req = {0};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(fd, VIDIOC_REQBUFS, &req);

    // Configuration du tampon
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    xioctl(fd, VIDIOC_QUERYBUF, &buf);

    Buffer videoBuffer;
    videoBuffer.start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (videoBuffer.start == MAP_FAILED) {
        perror("mmap");
        close(fd);
        exit(EXIT_FAILURE);
    }
    videoBuffer.length = buf.length;

    xioctl(fd, VIDIOC_QBUF, &buf);
    xioctl(fd, VIDIOC_STREAMON, &buf.type);

    xioctl(fd, VIDIOC_DQBUF, &buf);

    // Traitement de l'image
    unsigned char *imageData = videoBuffer.start;
    //displayImageWithSDL(imageData, 1280, 720);

    // Convertir et normaliser l'image
    for (int y = 0; y < INPUT_HEIGHT; ++y) {
        for (int x = 0; x < INPUT_WIDTH; ++x) {
            int srcX = x * 1280 / INPUT_WIDTH;
            int srcY = y * 720 / INPUT_HEIGHT;
            int srcIndex = (srcY * 1280 + srcX) * 3;
            int dstIndex = y * INPUT_WIDTH + x;

            // Convertir l'image en niveaux de gris
            unsigned char r = imageData[srcIndex + 0];
            unsigned char g = imageData[srcIndex + 1];
            unsigned char b = imageData[srcIndex + 2];
            double gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

            buffer[dstIndex] = gray;
        }
    }

    // Libération des ressources
    xioctl(fd, VIDIOC_STREAMOFF, &buf.type);
    munmap(videoBuffer.start, videoBuffer.length);
    close(fd);
}

// Fonction d'affichage de l'image avec SDL
void displayImageWithSDL(unsigned char *image, int width, int height) { //TODO : changer le type de l'image et la faire fonctionner
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Erreur SDL_Init: %s\n", SDL_GetError());
        exit(EXIT_FAILURE);
    }

    SDL_Window *window = SDL_CreateWindow("Image Capture", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "Erreur SDL_CreateWindow: %s\n", SDL_GetError());
        SDL_Quit();
        exit(EXIT_FAILURE);
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Erreur SDL_CreateRenderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        exit(EXIT_FAILURE);
    }

    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!texture) {
        fprintf(stderr, "Erreur SDL_CreateTexture: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        exit(EXIT_FAILURE);
    }

    SDL_UpdateTexture(texture, NULL, image, width * 3);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    SDL_Delay(50); // Affiche l'image pendant 50 millisecondes

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}



// Fonction pour charger et traiter l'image
void preprocess(unsigned char *imageData, int inputWidth, int inputHeight, double *buffer) {
    // Assumons que imageData contient l'image en format RGB24
    for (int y = 0; y < inputHeight; ++y) {
        for (int x = 0; x < inputWidth; ++x) {
            printf("x = %d, y = %d\n", x, y);
            // Calculer les indices source et destination
            int srcX = x * IMAGE_WIDTH / inputWidth;
            int srcY = y * IMAGE_HEIGHT / inputHeight;
            int srcIndex = (srcY * IMAGE_WIDTH + srcX) * 3;
            int dstIndex = y * inputWidth + x;

            // Extraire les composantes RGB
            unsigned char r = imageData[srcIndex + 0];
            unsigned char g = imageData[srcIndex + 1];
            unsigned char b = imageData[srcIndex + 2];

            // Convertir en niveaux de gris
            double gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            printf("gray = %f\n", gray);

            // Stocker le résultat dans le tampon de sortie
            buffer[dstIndex] = gray;
        }
    }
}

void image_resize(double* buffer_ancient, int newWidth, int newHeight, int origWidth, int origHeight, double *buffer) {
    
    int largeur = origWidth;
    int hauteur = origHeight;
    // Normalisation
    for (int i = 0; i < largeur * hauteur; i++) {
        buffer[i] = buffer_ancient[i] / 255.0;
    }

    // Redimensionnement
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int srcX = x * largeur / newWidth;
            int srcY = y * hauteur / newHeight;
            int srcIndex = srcY * largeur + srcX;
            int dstIndex = y * newWidth + x;
            buffer[dstIndex] = buffer[srcIndex];
        }
    }
}

#include <jpeglib.h>
#include <jerror.h>
#include <dirent.h>

// Fonction pour lire une image JPG et la convertir en double*
double* read_jpeg_image(const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE * infile;
    JSAMPARRAY buffer;
    int row_stride;
    double *image_data;
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    
    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return NULL;
    }
    
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    
    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int channels = cinfo.output_components;
    
    row_stride = cinfo.output_width * cinfo.output_components;
    image_data = (double*)malloc(sizeof(double) * (width) * (height) * (channels));
    
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    int i = 0;
    
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (int j = 0; j < row_stride; j++) {
            image_data[i++] = (double)buffer[0][j] / 255.0; // Normaliser les valeurs sur [0, 1]
        }
    }
    
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    return image_data;
}


// Fonction pour redimensionner une image
int remplisseur_buffer(const char *path, double *buffer) {
    DIR *dir;
    struct dirent *entry;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file;
    JSAMPARRAY buffer_ptr;
    int i, j;

    if ((dir = opendir(path)) == NULL) {
        perror("Unable to open directory");
        return 1;
    }

    i = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".jpg") != NULL) {
            char filepath[1024];
            sprintf(filepath, "%s/%s", path, entry->d_name);

            file = fopen(filepath, "rb");
            if (!file) {
                perror("Unable to open file");
                continue;
            }

            cinfo.err = jpeg_std_error(&jerr);
            jpeg_create_decompress(&cinfo);
            jpeg_stdio_src(&cinfo, file);
            jpeg_read_header(&cinfo, TRUE);
            jpeg_start_decompress(&cinfo);

            buffer_ptr = (*cinfo.mem->alloc_sarray)
                        ((j_common_ptr) &cinfo, JPOOL_IMAGE, cinfo.output_width * cinfo.output_components, 1);

            while (cinfo.output_scanline < cinfo.output_height) {
                jpeg_read_scanlines(&cinfo, buffer_ptr, 1);
                for (j = 0; j < cinfo.output_width * cinfo.output_components; j++) {
                    buffer[i++] = buffer_ptr[0][j];
                }
            }

            jpeg_finish_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            fclose(file);
        }
    }

    closedir(dir);
    return 0;
}

void chargeur_images_entrainement(const char *directoryPath, double *labels, int numImages) {
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
int main() {

    double *imageBuffer = (double *)malloc(INPUT_NODES * sizeof(double));
    if (imageBuffer == NULL) {
        perror("Allocation de mémoire échouée");
        exit(EXIT_FAILURE);
    }

    captureAndProcessImage(imageBuffer);
    //displayImageWithSDL(imageBuffer, INPUT_WIDTH, INPUT_HEIGHT);
    //int camWidth, camHeight;
    //Image img = captureImage(CAM_DEVICE, &camWidth, &camHeight);

    //unsigned char *resizedImage = resizeImage(img.buffer, camWidth, camHeight, INPUT_WIDTH, INPUT_HEIGHT);

    //double normalizedImage[INPUT_NODES * 3];
    //normalizeImage(resizedImage, INPUT_WIDTH, INPUT_HEIGHT, normalizedImage);

    NeuralNetwork nn;
    initNetwork(&nn);
    printf("Initialisation du réseau de neurones...\n");

    //double* entree = (double *)malloc((IMAGE_HEIGHT * IMAGE_WIDTH) * sizeof(double));
    //double input[IMAGE_HEIGHT * IMAGE_WIDTH]; // Entrée
    double *input = (double *)malloc((((IMAGE_HEIGHT * IMAGE_WIDTH) * 3 )* 2000) * sizeof(double));
    double targetOutput[OUTPUT_NODES]; // Sortie cible
    const char *directory_path = "./photoss_visage";//repertoire contenant les images
    remplisseur_buffer(directory_path, input);
    chargeur_images_entrainement(directory_path, targetOutput, 20);

    //for (int i = 0; i < 1000; ++i) {
    //}
    int epoch = 30;
    trainNetwork(&nn, input, targetOutput, epoch);

    double hiddenLayer1Output[HIDDEN_LAYER1_NODES];
    double hiddenLayer2Output[HIDDEN_LAYER2_NODES];
    double outputLayerOutput[OUTPUT_NODES];
    while(1) {
        captureAndProcessImage(imageBuffer);
        //displayImageWithSDL(imageBuffer, INPUT_WIDTH, INPUT_HEIGHT);
        feedForward(&nn, imageBuffer, hiddenLayer1Output, hiddenLayer2Output, outputLayerOutput);
        printf("Prédiction : visage: %f, %f\n", outputLayerOutput[0], outputLayerOutput[1]);
        if (outputLayerOutput[0] > outputLayerOutput[1]) {
            printf("Visage détecté\n");
        } else {
            printf("Pas de visage\n");
        }
    }
    //feedForward(&nn, imageBuffer, hiddenLayer1Output, hiddenLayer2Output, outputLayerOutput);
    //printf("Prédiction : visage: %f, %f\n", outputLayerOutput[0], outputLayerOutput[1]);



    freeNetwork(&nn);
    free(imageBuffer);
    //free(img.buffer);
    //free(resizedImage);

    return 0;
}