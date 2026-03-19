#include <stdio.h>
#include <windows.h>
#include <iostream>
#include <winbase.h>
using namespace std;

void SubThread(void){
    int i;
    for(i=0;i<5;i++){
        cout<<"SubThread"<<i<<endl;
        Sleep(2000);
    }
}

int main(void){
    cout<<"CreatThread"<<endl;

    //Creat a thread;
    DWORD IDThread;
    HANDLE hThread;
    hThread=CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)SubThread,NULL,0,&IDThread);
        //NULL,no security attributes
        //0,use default stack size
        //(LPTHREAD_START_ROUTINE)SubThread,thread function 
        //NULL,no thread function argument
        //0, use default creation flags 
        //&IDThread, returns thread identifier 

	// Check the return value for success. 
	if (hThread == NULL)
		cout << "CreateThread error" << endl;

    int i;
	for (i=0;i<5;i++){
        cout << "MainThread" << i << endl;
		if (i==1){
            if (SuspendThread(hThread)==0xFFFFFFFF){
                cout << "Suspend thread error." << endl;
            }
			else{
                cout << "Suspend thread is ok." << endl;
			}
		}
    }

    if (i==3){
        if (ResumeThread(hThread)==0xFFFFFFFF){
            cout << "Resume thread error." << endl;
        }
		else{
            cout << "Resume thread is ok." << endl;
        }
    }
    Sleep(4000);
    return 0;
}