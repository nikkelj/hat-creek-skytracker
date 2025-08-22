import pygame
import os
import sys
import webbrowser

if __name__ == "__main__":
    os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
    pygame.init()
    display_info = pygame.display.Info()
    menu_width = 200
    total_width = display_info.current_w
    total_height = display_info.current_h
    menu_screen = pygame.display.set_mode((total_width, total_height))
    pygame.display.set_caption("Main Menu")

    # Load background image for menu (assume 'cli/lucky.jpg' exists)
    try:
        bg_image = pygame.image.load('cli/lucky.jpg')
        bg_image = pygame.transform.scale(bg_image, (80, 80))  # Shrink to 80x80
    except pygame.error:
        bg_image = None  # Fallback to solid color if image not found
        print("Warning: 'cli/lucky.jpg' not found. Using fallback color.")

    font = pygame.font.Font(None, 24)
    large_font = pygame.font.Font(None, 36)

    sub_x = menu_width
    sub_y = 0
    sub_width = total_width - menu_width
    sub_height = total_height

    buttons = [
        {"rect": pygame.Rect(10, 10, 180, 80), "text": "Tracking Vis", "mode": "tracking_vis"},
        {"rect": pygame.Rect(10, 100, 180, 80), "text": "Sensor Calib", "mode": "sensor_calib"},
        {"rect": pygame.Rect(10, 190, 180, 80), "text": "Joystick Loop", "mode": "joystick_loop"},
        {"rect": pygame.Rect(10, 280, 180, 80), "text": "Post Process", "mode": "post_process"},
        {"rect": pygame.Rect(10, 370, 180, 80), "text": "Config Options", "mode": "config_options"},
        {"rect": pygame.Rect(10, 460, 180, 80), "text": "Author Info", "mode": "author_info"},
        {"rect": pygame.Rect(10, 550, 180, 80), "text": "Exit", "mode": "exit"},
    ]

    image_y = 550 + 80 + 10  # Position underneath the buttons

    current_mode = None
    clock = pygame.time.Clock()

    # For author info
    author_bg = None
    try:
        author_bg = pygame.image.load('cli/lucky.jpg')
    except pygame.error:
        pass

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for btn in buttons:
                    if btn["rect"].collidepoint(pos):
                        if btn["mode"] == "exit":
                            running = False
                        else:
                            current_mode = btn["mode"]
                if current_mode:
                    if current_mode == "author_info":
                        contact_text = "Jonathan Nikkel - @nikkeljon"
                        text2 = large_font.render(contact_text, True, (255, 255, 255))
                        text_rect = text2.get_rect(topleft=(sub_x + 10, sub_y + 50))
                        if text_rect.collidepoint(pos):
                            webbrowser.open("https://x.com/NikkelJonathan")

        menu_screen.fill((200, 200, 200), (0, 0, menu_width, total_height))  # Menu background

        if bg_image:
            menu_screen.blit(bg_image, ((menu_width - 80) // 2, image_y))  # Center horizontally underneath buttons

        for btn in buttons:
            pygame.draw.rect(menu_screen, (211, 211, 211), btn["rect"])  # Light grey button color
            text = font.render(btn["text"], True, (0, 0, 0))
            menu_screen.blit(text, (btn["rect"].x + 10, btn["rect"].y + 30))

        if current_mode:
            sub_rect = (sub_x, sub_y, sub_width, sub_height)
            if current_mode == "tracking_vis":
                menu_screen.fill((0, 0, 0), sub_rect)
                # Add tracking visualization code here later
            elif current_mode == "sensor_calib":
                menu_screen.fill((50, 50, 50), sub_rect)
                # Add sensor-mount calibration code here later
            elif current_mode == "joystick_loop":
                menu_screen.fill((100, 100, 100), sub_rect)
                # Add manual joystick loop code here later
            elif current_mode == "post_process":
                menu_screen.fill((150, 150, 150), sub_rect)
                # Add post-processing tool code here later
            elif current_mode == "config_options":
                menu_screen.fill((200, 200, 0), sub_rect)
                # Add configuration options code here later
            elif current_mode == "author_info":
                if author_bg:
                    scaled_bg = pygame.transform.scale(author_bg, (sub_width, sub_height))
                    menu_screen.blit(scaled_bg, (sub_x, sub_y))
                else:
                    menu_screen.fill((0, 0, 0), sub_rect)
                text1 = large_font.render("Starlink-1060", True, (255, 255, 255))
                menu_screen.blit(text1, (sub_x + 10, sub_y + 10))
                contact_text = "Jonathan Nikkel - @nikkeljon"
                text2 = large_font.render(contact_text, True, (255, 255, 255))
                menu_screen.blit(text2, (sub_x + 10, sub_y + 50))

        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS for better responsiveness

    pygame.quit()