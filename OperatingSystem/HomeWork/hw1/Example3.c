#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/wait.h>
#include <errno.h>
#include <stdlib.h>

pid_t wait(int* stat_loc);
void perror(const char *s);

int errno;
int global;

void main(){
    int local=0,i;
    pid_t child;
    if((child=fork())==-1){//创建失败
        printf("Fork Error.\n");
    }
    if(child==0){//子进程
        printf("Now it is in child process.\n");
        if(execl("/home/aye/Desktop/New Folder/test","test",NULL)==-1){
            perror("Error in child process");//加载失败
        }
        global=local+2;
        exit(0);
    }
    //父进程
    printf("Now it is in parent process.\n");
    for(i=0;i<10;i++){
        sleep(2);
        printf("Parent loop:%d\n",i);
        if(i==2){
            if((child=fork())==-1){
                //创建失败
                printf("Fork Error.\n");
            }
            if(child==0){
                //子进程
                printf("Now it is in child process.\n");
                if(execl("/home/aye/Desktop/New Folder/test","test",NULL)==-1){//加载程序失败
                    perror("Error in child process");
                }
                global=local+2;
                exit(0);
            }
        }
        if(i==3){
            pid_t temp;
            temp=wait(NULL);
            printf("Child process ID:%d\n",temp);
        }
    }
    global=local+1;
    printf("Parent process is end,the local is %d,the global is %d.\n",local,global);
    exit(0);
}
