#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

int main(void){
    int pid;
    printf("Just 1 process now.\n");
    printf("Parent process calling fork()......\n");
    pid=fork();
    if(pid==0){
        printf("I am the child.\n");
        execl("/bin/ls","ls","-l","test.c",NULL);
        perror("exec error:");
        exit(0);
    }
    else if(pid>0){
        wait(NULL);
        printf("I am the parent.\n");
    }
    else
        printf("fork failed.\n");
    printf("program end.\n");
}