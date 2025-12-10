#include <iostream>
#include <winsock2.h>
#include <chrono>
#include <thread>
#include <vector>
#include <sstream>
#include <string>


using namespace std;

#pragma comment(lib,"ws2_32.lib")
#pragma warning(disable:4996)

#include <HD/hd.h>
#include <HDU/hduError.h>
#include <HDU/hduVector.h>

#define SERVER "127.0.0.1"  // or "localhost" - ip address of UDP server
#define BUFLEN 512  // max length of answer
#define PORT 12312  // the port on which to listen for incoming data
bool print_message = false;

// Teleoperation variables
bool isTeleoperationOn = true;
std::vector<double> master_position = { 0.0, 0.0, 0.0 };
std::vector<double> master_position_init = { 0.0, 0.0, 0.0 };
std::vector<double> master_force = { 0.0, 0.0, 0.0 };
int master_timestamp = 0;
int master_button_click_times = 0;
bool master_button_JustClicked = false;
int button_click = 0;
bool recenter = true;

bool timestamp_saved = false;
int warm_up_count = 0;
int warm_up_epoch = 5;

std::vector<double> position_aver = { 0.0, 0.0, 0.0 };
std::vector<double> velocity_aver = { 0.0, 0.0, 0.0 };

HDCallbackCode HDCALLBACK master_interface(void* data)
{
	HDErrorInfo error;
	hduVector3Dd position;
	hduVector3Dd rotation;
	hduVector3Dd button;
	HDdouble trans[16];
	hduVector3Dd angles;

	//hduVector3Dd force;

	HHD hHD = hdGetCurrentDevice();

	hdBeginFrame(hHD);
	hdGetDoublev(HD_CURRENT_POSITION, position);
	printf("pos: %.2f  %.2f  %.2f\n", position[0], position[1], position[2]);
	hdGetDoublev(HD_CURRENT_TRANSFORM, trans);
	hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, angles);
	hdGetDoublev(HD_CURRENT_BUTTONS, button);

	HDfloat baseForce[3];
	float force_multiplier = 0.02;
	// float center[3] = { 0.0, 20.0, 30.0 };  // should be in cm (0.0, 20.0, 40.0)
	float center[3] = { 0.0, 0.0, 0.0 };
	button_click = (int)button[0];
	if (button_click)
	{
		if (!master_button_JustClicked)  // a new click
		{
			master_button_click_times++;
			master_button_JustClicked = true;
		}
	}
	else
		master_button_JustClicked = false;

	if (master_button_click_times == 0)  // initial
		recenter = true;
	else  // rendering
		recenter = false;

	if (recenter == true)
	{
		for (int i = 0; i <= 2; i++)
		{
			master_position[i] = position[i] - center[i];  // use center as origin
			baseForce[i] = -master_position[i] * force_multiplier;  // back to center
		}
		hdSetFloatv(HD_CURRENT_FORCE, baseForce);
	}
	else  // rendering
	{
		for (int i = 0; i <= 2; i++)
		{
			master_position[i] = position[i] - master_position_init[i];  // use user's first position as origin
			baseForce[i] = master_force[i];  // force feedback
		}
		hdSetFloatv(HD_CURRENT_FORCE, baseForce);
	}

	master_timestamp = (master_timestamp + 1);

	hdEndFrame(hHD);

	if (HD_DEVICE_ERROR(error = hdGetError()))
	{
		hduPrintError(stderr, &error, "Error detected while rendering gravity well\n");

		if (hduIsSchedulerError(&error))
		{
			return HD_CALLBACK_DONE;
		}
	}

	return HD_CALLBACK_CONTINUE;
}

int main()
{

	// Haptic Device

	// Initializing haptic device
	HDErrorInfo error_hapt;
	HDSchedulerHandle hMasterSite;


	HHD hHD = hdInitDevice(HD_DEFAULT_DEVICE);
	if (HD_DEVICE_ERROR(error_hapt = hdGetError()))
	{
		hduPrintError(stderr, &error_hapt, "Failed to initialize haptic device");
		fprintf(stderr, "\nPress any key to quit.\n");
		system("pause");
		return -1;
	}

	printf("Found haptic device model: %s.\n\n", hdGetString(HD_DEVICE_MODEL_TYPE));

	/* Schedule the main callback that will render forces to the device. */
	hMasterSite = hdScheduleAsynchronous(master_interface, 0, HD_MAX_SCHEDULER_PRIORITY);

	hdEnable(HD_FORCE_OUTPUT);
	hdStartScheduler();

	/* Check for errors and abort if so. */
	if (HD_DEVICE_ERROR(error_hapt = hdGetError()))
	{
		hduPrintError(stderr, &error_hapt, "Failed to start scheduler");
		fprintf(stderr, "\nPress any key to quit.\n");
		return -1;
	}

	system("title UDP Client");

	// initialise winsock
	WSADATA ws;
	printf("Initialising Winsock...");
	if (WSAStartup(MAKEWORD(2, 2), &ws) != 0)
	{
		printf("Failed. Error Code: %d", WSAGetLastError());
		return 1;
	}
	printf("Initialised.\n");

	// create socket
	sockaddr_in server;
	int client_socket;
	if ((client_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == SOCKET_ERROR) // <<< UDP socket
	{
		printf("socket() failed with error code: %d", WSAGetLastError());
		return 2;
	}

	// setup address structure
	memset((char*)&server, 0, sizeof(server));
	server.sin_family = AF_INET;
	server.sin_port = htons(PORT);
	server.sin_addr.S_un.S_addr = inet_addr(SERVER);

	std::cout << "Haptic rendering started :)" << std::endl;
	// start communication
	while (true)
	{
		if (master_button_click_times == 1)  // start rendering
		{
			// first send some warm up info to avoid slow update at beginning
			if (warm_up_count >= warm_up_epoch)  // already warmed up
			{
				if (master_timestamp != timestamp_saved)
					timestamp_saved = master_timestamp;
				else
					continue;

				std::stringstream msg_strea;
				msg_strea << "{"
					<< "\"position\":[" << master_position[0] << ", " << master_position[1] << ", " << master_position[2] << "],"
					<< "\"timestamp\":" << master_timestamp
					<< "}";
				string msg_str = msg_strea.str();
				if (sendto(client_socket, msg_str.c_str(), strlen(msg_str.c_str()), 0, (sockaddr*)&server, sizeof(sockaddr_in)) == SOCKET_ERROR)
				{
					printf("sendto() failed with error code: %d", WSAGetLastError());
					return 3;
				}
				if (print_message)	std::cout << "Sent: " << msg_str << std::endl;

				char answer[BUFLEN] = {};
				int slen = sizeof(sockaddr_in);
				int answer_length;
				answer_length = recvfrom(client_socket, answer, BUFLEN, 0, (sockaddr*)&server, &slen);
				if (answer_length > 0)
				{
					if (print_message)	std::cout << "Received: " << answer << std::endl;
					char* info_keyword_ptr = strstr(answer, "\"force\"");
					char info_data[BUFLEN] = {};
					if (info_keyword_ptr != nullptr)
					{
						char* info_start_ptr = strstr(info_keyword_ptr, "[");
						info_start_ptr++;
						char* info_end_ptr = strstr(info_start_ptr, ",");
						int info_length = info_end_ptr - info_start_ptr;
						strncpy(info_data, info_start_ptr, info_length);
						master_force[0] = std::stod(info_data);
						memset(info_data, 0, sizeof(info_data));

						info_start_ptr = info_end_ptr + 1;
						info_end_ptr = strstr(info_start_ptr, ",");
						info_length = info_end_ptr - info_start_ptr;
						strncpy(info_data, info_start_ptr, info_length);
						master_force[1] = std::stod(info_data);
						memset(info_data, 0, sizeof(info_data));

						info_start_ptr = info_end_ptr + 1;
						info_end_ptr = strstr(info_start_ptr, "]");
						info_length = info_end_ptr - info_start_ptr;
						strncpy(info_data, info_start_ptr, info_length);
						master_force[2] = std::stod(info_data);
						memset(info_data, 0, sizeof(info_data));

						if (print_message)	std::cout << "Extracted info: " << master_force[0] << "," << master_force[1] << "," << master_force[2] << "\n" << std::endl;
					}
				}
			}
			else  // still warming up -> send 0 position
			{
				if (master_timestamp != timestamp_saved)
				{
					timestamp_saved = master_timestamp;
					// start from user's first position
					if (warm_up_count == warm_up_epoch - 1)
						master_position_init = master_position;
				}
				else
					continue;

				//char message[BUFLEN];
				//cin.getline(message, BUFLEN);
				std::stringstream msg_strea;
				msg_strea << "{"
					<< "\"position\":[" << 0.0 << ", " << 0.0 << ", " << 0.0 << "],"
					<< "\"velocity\":[" << 0.0 << ", " << 0.0 << ", " << 0.0 << "],"
					<< "\"timestamp\":" << master_timestamp
					<< "}";
				string msg_str = msg_strea.str();
				// send the message
				if (sendto(client_socket, msg_str.c_str(), strlen(msg_str.c_str()), 0, (sockaddr*)&server, sizeof(sockaddr_in)) == SOCKET_ERROR)
				{
					printf("sendto() failed with error code: %d", WSAGetLastError());
					return 3;
				}
				if (print_message)	std::cout << "Sent: " << msg_str << std::endl;

				//this_thread::sleep_for(std::chrono::milliseconds(20));

				//receive a reply and print it
				//clear the answer by filling null, it might have previously received data
				char answer[BUFLEN] = {};
				// try to receive some data, this is a blocking call
				int slen = sizeof(sockaddr_in);
				int answer_length;
				/*if (answer_length = recvfrom(client_socket, answer, BUFLEN, 0, (sockaddr*)&server, &slen) == SOCKET_ERROR)
				{
					printf("recvfrom() failed with error code: %d", WSAGetLastError());
					exit(0);
				}*/
				answer_length = recvfrom(client_socket, answer, BUFLEN, 0, (sockaddr*)&server, &slen);
				if (print_message && answer_length > 0)
					std::cout << "Received: " << answer << std::endl;

				warm_up_count++;
			}
		}
		else if (master_button_click_times >= 2)  // stop flag
		{
			if (warm_up_count >= warm_up_epoch)  // this needs to be after the warm up process
			{
				//char message[BUFLEN];
				//cin.getline(message, BUFLEN);
				std::stringstream msg_strea;
				msg_strea << "{"
					<< "\"position\":[" << position_aver[0] << ", " << position_aver[1] << ", " << position_aver[2] << "],"
					<< "\"velocity\":[" << velocity_aver[0] << ", " << velocity_aver[1] << ", " << velocity_aver[2] << "],"
					<< "\"timestamp\":" << "-1.0"
					<< "}";
				string msg_str = msg_strea.str();
				// send the message
				if (sendto(client_socket, msg_str.c_str(), strlen(msg_str.c_str()), 0, (sockaddr*)&server, sizeof(sockaddr_in)) == SOCKET_ERROR)
				{
					printf("sendto() failed with error code: %d", WSAGetLastError());
					return 3;
				}

				std::cout << "Stop and restart! (timestamp: " << master_timestamp << ")" << std::endl;
				for (int i = 0; i < 3; i++)
				{
					master_position[i] = 0.0;
					master_position_init[i] = 0.0;
					master_force[i] = 0.0;
				}
				warm_up_count = 0;
				master_button_click_times = 0;
				master_timestamp = 0;
			}
		}
	}
	closesocket(client_socket);
	WSACleanup();
}
