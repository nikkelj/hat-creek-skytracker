import math
import pygame
from datetime import datetime, timedelta
from utils import get_altitude_color, draw_button

def draw_hexagon(surface, x, y, color, size=5):
    points = [(x + size * math.cos(math.radians(60 * i)), y + size * math.sin(math.radians(60 * i))) for i in range(6)]
    pygame.draw.polygon(surface, color, points)

def draw_triangle(surface, x, y, color, size=5):
    points = [(x, y - size), (x - size * math.sin(math.radians(60)), y + size * 0.5), (x + size * math.sin(math.radians(60)), y + size * 0.5)]
    pygame.draw.polygon(surface, color, points)

def draw_polar_plot(surface, sub_x, sub_y, sub_width, sub_height, input_rects, lat_str, lon_str, alt_str, elevation_mask_str, font, focused_field, cursor_pos, selection_start, save_button, load_button, button_states, ts=None, current_tt=None, satellite_trajectories=None, satellite_positions=None, satellite_labels=None, satellite_mean_altitudes=None, selected_satellite=None, satellite_arc_segments=None, filter_text="", filter_above_alt_text="", legend_x=None, legend_y=None, clear_filters_button=None, tle_loaded=False):
    # Draw gradient background for config options
    if ts is None:
        for y in range(sub_height):
            color = (160 - (y / sub_height * 5), 160 - (y / sub_height * 5), 160 - (y / sub_height * 5))
            pygame.draw.line(surface, color, (sub_x, sub_y + y), (sub_x + sub_width, sub_y + y))
        # Draw grouping box and label
        group_rect = pygame.Rect(sub_x + 10, sub_y + 0, 220, 370)
        pygame.draw.rect(surface, (0, 0, 0), group_rect, 2, border_radius=5)
        group_label = font.render("Observer Location", True, (0, 0, 0))
        surface.blit(group_label, (sub_x + 20, sub_y + 10))
        # Draw labels and inputs
        lat_label = font.render("Latitude:", True, (0, 0, 0))
        surface.blit(lat_label, (sub_x + 20, sub_y + 30))
        pygame.draw.rect(surface, (255, 255, 255), input_rects['lat'])
        lat_text = font.render(lat_str, True, (0, 0, 0))
        surface.blit(lat_text, (input_rects['lat'].x + 5, input_rects['lat'].y + 5))
        if focused_field == 'lat':
            pygame.draw.rect(surface, (0, 0, 255), input_rects['lat'], 2)
            text_width, _ = font.size(lat_str[:cursor_pos['lat']])
            pygame.draw.line(surface, (0, 0, 255),
                            (input_rects['lat'].x + 5 + text_width, input_rects['lat'].y + 5),
                            (input_rects['lat'].x + 5 + text_width, input_rects['lat'].y + 25), 2)
            if selection_start['lat'] is not None:
                start_width, _ = font.size(lat_str[:min(cursor_pos['lat'], selection_start['lat'])])
                end_width, _ = font.size(lat_str[:max(cursor_pos['lat'], selection_start['lat'])])
                pygame.draw.rect(surface, (0, 120, 215),
                                (input_rects['lat'].x + 5 + start_width, input_rects['lat'].y + 5,
                                 end_width - start_width, 20), 2)

        lon_label = font.render("Longitude:", True, (0, 0, 0))
        surface.blit(lon_label, (sub_x + 20, sub_y + 120))
        pygame.draw.rect(surface, (255, 255, 255), input_rects['lon'])
        lon_text = font.render(lon_str, True, (0, 0, 0))
        surface.blit(lon_text, (input_rects['lon'].x + 5, input_rects['lon'].y + 5))
        if focused_field == 'lon':
            pygame.draw.rect(surface, (0, 0, 255), input_rects['lon'], 2)
            text_width, _ = font.size(lon_str[:cursor_pos['lon']])
            pygame.draw.line(surface, (0, 0, 255),
                            (input_rects['lon'].x + 5 + text_width, input_rects['lon'].y + 5),
                            (input_rects['lon'].x + 5 + text_width, input_rects['lon'].y + 25), 2)
            if selection_start['lon'] is not None:
                start_width, _ = font.size(lon_str[:min(cursor_pos['lon'], selection_start['lon'])])
                end_width, _ = font.size(lon_str[:max(cursor_pos['lon'], selection_start['lon'])])
                pygame.draw.rect(surface, (0, 120, 215),
                                (input_rects['lon'].x + 5 + start_width, input_rects['lon'].y + 5,
                                 end_width - start_width, 20), 2)

        alt_label = font.render("Altitude (m):", True, (0, 0, 0))
        surface.blit(alt_label, (sub_x + 20, sub_y + 210))
        pygame.draw.rect(surface, (255, 255, 255), input_rects['alt'])
        alt_text = font.render(alt_str, True, (0, 0, 0))
        surface.blit(alt_text, (input_rects['alt'].x + 5, input_rects['alt'].y + 5))
        if focused_field == 'alt':
            pygame.draw.rect(surface, (0, 0, 255), input_rects['alt'], 2)
            text_width, _ = font.size(alt_str[:cursor_pos['alt']])
            pygame.draw.line(surface, (0, 0, 255),
                            (input_rects['alt'].x + 5 + text_width, input_rects['alt'].y + 5),
                            (input_rects['alt'].x + 5 + text_width, input_rects['alt'].y + 25), 2)
            if selection_start['alt'] is not None:
                start_width, _ = font.size(alt_str[:min(cursor_pos['alt'], selection_start['alt'])])
                end_width, _ = font.size(alt_str[:max(cursor_pos['alt'], selection_start['alt'])])
                pygame.draw.rect(surface, (0, 120, 215),
                                (input_rects['alt'].x + 5 + start_width, input_rects['alt'].y + 5,
                                 end_width - start_width, 20), 2)

        elevation_mask_label = font.render("Elevation Mask (deg):", True, (0, 0, 0))
        surface.blit(elevation_mask_label, (sub_x + 20, sub_y + 300))
        pygame.draw.rect(surface, (255, 255, 255), input_rects['elevation_mask'])
        elevation_mask_text = font.render(elevation_mask_str, True, (0, 0, 0))
        surface.blit(elevation_mask_text, (input_rects['elevation_mask'].x + 5, input_rects['elevation_mask'].y + 5))
        if focused_field == 'elevation_mask':
            pygame.draw.rect(surface, (0, 0, 255), input_rects['elevation_mask'], 2)
            text_width, _ = font.size(elevation_mask_str[:cursor_pos['elevation_mask']])
            pygame.draw.line(surface, (0, 0, 255),
                            (input_rects['elevation_mask'].x + 5 + text_width, input_rects['elevation_mask'].y + 5),
                            (input_rects['elevation_mask'].x + 5 + text_width, input_rects['elevation_mask'].y + 25), 2)
            if selection_start['elevation_mask'] is not None:
                start_width, _ = font.size(elevation_mask_str[:min(cursor_pos['elevation_mask'], selection_start['elevation_mask'])])
                end_width, _ = font.size(elevation_mask_str[:max(cursor_pos['elevation_mask'], selection_start['elevation_mask'])])
                pygame.draw.rect(surface, (0, 120, 215),
                                (input_rects['elevation_mask'].x + 5 + start_width, input_rects['elevation_mask'].y + 5,
                                 end_width - start_width, 20), 2)

        # Draw buttons and divider
        pygame.draw.line(surface, (0, 0, 0), (sub_x, sub_y + sub_height - 60), (sub_x + sub_width, sub_y + sub_height - 60), 1)
        draw_button(surface, save_button, "Save", button_states["save"])
        draw_button(surface, load_button, "Load", button_states["load"])
    else:
        elevation_mask = float(elevation_mask_str or 0)
        # Draw polar plot for tracking vis
        cx = sub_x + sub_width // 2
        cy = sub_y + sub_height // 2
        radius = min(sub_width, sub_height) // 2 - 50
        # Draw horizon circle
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), radius, 1)
        # Draw elevation mask circle
        mask_radius = (90 - elevation_mask) / 90 * radius
        pygame.draw.circle(surface, (255, 0, 0), (cx, cy), mask_radius, 2)
        # Draw elevation circles
        for el in [30, 60]:
            r = (90 - el) / 90 * radius
            pygame.draw.circle(surface, (100, 100, 100), (cx, cy), int(r), 1)
            el_label = pygame.font.Font(None, 14).render(f"{el}°", True, (255, 255, 255))
            surface.blit(el_label, (cx + r + 5, cy - 5))
        # Draw azimuth lines and labels
        for az_deg in range(0, 360, 30):
            az_rad = math.radians(az_deg)
            x1 = cx + radius * math.sin(az_rad)
            y1 = cy - radius * math.cos(az_rad)
            pygame.draw.line(surface, (100, 100, 100), (cx, cy), (x1, y1), 1)
            if az_deg % 90 == 0:
                direction = {0: "N", 90: "E", 180: "S", 270: "W"}[az_deg]
                direction_label = pygame.font.Font(None, 14).render(direction, True, (255, 255, 255))
                if az_deg == 0:  # North
                    surface.blit(direction_label, (cx - direction_label.get_width() // 2, cy - radius - 10))
                elif az_deg == 90:  # East
                    surface.blit(direction_label, (cx + radius + 10, cy - direction_label.get_height() // 2))
                elif az_deg == 180:  # South
                    surface.blit(direction_label, (cx - direction_label.get_width() // 2, cy + radius + 10))
                elif az_deg == 270:  # West
                    surface.blit(direction_label, (cx - radius - 10 - direction_label.get_width(), cy - direction_label.get_height() // 2))
        # Draw precomputed arc segments for selected satellite
        if selected_satellite and tle_loaded and selected_satellite in satellite_arc_segments:
            for x0, y0, x1, y1, color in satellite_arc_segments[selected_satellite]:
                pygame.draw.line(surface, color, (x0, y0), (x1, y1), 1)

def draw_satellites(surface, satellite_positions, satellite_labels, satellite_mean_altitudes, hovered_satellite, selected_satellite, cx, cy):
    for sat, (px, py, alt, _) in satellite_positions.items():
        mean_altitude = satellite_mean_altitudes.get(sat, 0.0)
        eccentricity = sat.model.ecco
        if 2000 < mean_altitude <= 35786:  # MEO
            color = (255, 165, 0)  # Orange
            draw_hexagon(surface, px, py, color)
        elif abs(mean_altitude - 35786) <= 1000:  # GEO or nearby
            color = (128, 0, 128)  # Purple
            draw_triangle(surface, px, py, color)
        else:  # LEO (0-2000 km)
            color = get_altitude_color(mean_altitude) or (0, 255, 0)  # Fallback to green if out of range
            if eccentricity > 0.01:
                width = 6
                height = 3
                angle = math.degrees(math.atan2(py - cy, px - cx))
                oval_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                pygame.draw.ellipse(oval_surface, color, (0, 0, width, height))
                rotated_oval = pygame.transform.rotate(oval_surface, angle)
                rotated_rect = rotated_oval.get_rect(center=(px, py))
                surface.blit(rotated_oval, rotated_rect.topleft)
            else:
                pygame.draw.circle(surface, color, (px, py), 3)
        if sat == hovered_satellite or sat == selected_satellite:
            pygame.draw.circle(surface, (255, 255, 0), (px, py), 5, 1)  # Highlight on hover or select
        surface.blit(satellite_labels[sat], (px + 5, py))

def draw_filters(surface, filter_rect, filter_above_alt_rect, filter_below_alt_rect, filter_text, filter_above_alt_text, filter_below_alt_text, focused_field, cursor_pos, selection_start, small_font):
    # Name filter
    filter_label = small_font.render("Name Filter:", True, (255, 255, 255))
    surface.blit(filter_label, (filter_rect.x, filter_rect.y - filter_label.get_height() - 5))
    pygame.draw.rect(surface, (255, 255, 255), filter_rect)
    filter_text_surface = small_font.render(filter_text, True, (0, 0, 0))
    surface.blit(filter_text_surface, (filter_rect.x + 5, filter_rect.y + 5))
    if focused_field == "filter":
        pygame.draw.rect(surface, (0, 0, 255), filter_rect, 2)
        text_width, _ = small_font.size(filter_text[:cursor_pos["filter"]])
        pygame.draw.line(surface, (0, 0, 255),
                        (filter_rect.x + 5 + text_width, filter_rect.y + 5),
                        (filter_rect.x + 5 + text_width, filter_rect.y + 25), 2)
        if selection_start["filter"] is not None:
            start_width, _ = small_font.size(filter_text[:min(cursor_pos["filter"], selection_start["filter"])])
            end_width, _ = small_font.size(filter_text[:max(cursor_pos["filter"], selection_start["filter"])])
            pygame.draw.rect(surface, (0, 120, 215),
                            (filter_rect.x + 5 + start_width, filter_rect.y + 5,
                             end_width - start_width, 20), 2)

    # Filter Above Altitude
    filter_above_alt_label = small_font.render("Filter Above Alt (km):", True, (255, 255, 255))
    surface.blit(filter_above_alt_label, (filter_above_alt_rect.x, filter_above_alt_rect.y - filter_above_alt_label.get_height() - 5))
    pygame.draw.rect(surface, (255, 255, 255), filter_above_alt_rect)
    filter_above_alt_text_surface = small_font.render(filter_above_alt_text, True, (0, 0, 0))
    surface.blit(filter_above_alt_text_surface, (filter_above_alt_rect.x + 5, filter_above_alt_rect.y + 5))
    if focused_field == "filter_above_alt":
        pygame.draw.rect(surface, (0, 0, 255), filter_above_alt_rect, 2)
        text_width, _ = small_font.size(filter_above_alt_text[:cursor_pos["filter_above_alt"]])
        pygame.draw.line(surface, (0, 0, 255),
                        (filter_above_alt_rect.x + 5 + text_width, filter_above_alt_rect.y + 5),
                        (filter_above_alt_rect.x + 5 + text_width, filter_above_alt_rect.y + 25), 2)
        if selection_start["filter_above_alt"] is not None:
            start_width, _ = small_font.size(filter_above_alt_text[:min(cursor_pos["filter_above_alt"], selection_start["filter_above_alt"])])
            end_width, _ = small_font.size(filter_above_alt_text[:max(cursor_pos["filter_above_alt"], selection_start["filter_above_alt"])])
            pygame.draw.rect(surface, (0, 120, 215),
                            (filter_above_alt_rect.x + 5 + start_width, filter_above_alt_rect.y + 5,
                             end_width - start_width, 20), 2)

    # Filter Below Altitude
    filter_below_alt_label = small_font.render("Filter Below Alt (km):", True, (255, 255, 255))
    surface.blit(filter_below_alt_label, (filter_below_alt_rect.x, filter_below_alt_rect.y - filter_below_alt_label.get_height() - 5))
    pygame.draw.rect(surface, (255, 255, 255), filter_below_alt_rect)
    filter_below_alt_text_surface = small_font.render(filter_below_alt_text, True, (0, 0, 0))
    surface.blit(filter_below_alt_text_surface, (filter_below_alt_rect.x + 5, filter_below_alt_rect.y + 5))
    if focused_field == "filter_below_alt":
        pygame.draw.rect(surface, (0, 0, 255), filter_below_alt_rect, 2)
        text_width, _ = small_font.size(filter_below_alt_text[:cursor_pos["filter_below_alt"]])
        pygame.draw.line(surface, (0, 0, 255),
                        (filter_below_alt_rect.x + 5 + text_width, filter_below_alt_rect.y + 5),
                        (filter_below_alt_rect.x + 5 + text_width, filter_below_alt_rect.y + 25), 2)
        if selection_start["filter_below_alt"] is not None:
            start_width, _ = small_font.size(filter_below_alt_text[:min(cursor_pos["filter_below_alt"], selection_start["filter_below_alt"])])
            end_width, _ = small_font.size(filter_below_alt_text[:max(cursor_pos["filter_below_alt"], selection_start["filter_below_alt"])])
            pygame.draw.rect(surface, (0, 120, 215),
                            (filter_below_alt_rect.x + 5 + start_width, filter_below_alt_rect.y + 5,
                             end_width - start_width, 20), 2)

def draw_legend(surface, legend_x, legend_y, small_font):
    pygame.draw.rect(surface, (50, 50, 50), (legend_x, legend_y, 150, 140))  # Larger legend
    pygame.draw.line(surface, (255, 255, 255), (legend_x, legend_y + 20), (legend_x + 150, legend_y + 20), 1)  # Title line
    legend_title = small_font.render("Orbit Legend", True, (255, 255, 255))
    surface.blit(legend_title, (legend_x + 5, legend_y + 5))
    # LEO heatmap
    for i in range(150):
        alt = i / 150 * 1000  # Scale from 0 to 1000 km
        color = get_altitude_color(alt) or (0, 255, 0)  # Fallback to green if out of range
        pygame.draw.line(surface, color, (legend_x + i, legend_y + 40), (legend_x + i, legend_y + 60))
    low_alt = small_font.render("0 km", True, (255, 255, 255))
    high_alt = small_font.render("1000 km", True, (255, 255, 255))
    surface.blit(low_alt, (legend_x, legend_y + 70))
    surface.blit(high_alt, (legend_x + 140, legend_y + 70))
    # MEO hexagon
    draw_hexagon(surface, legend_x + 20, legend_y + 90, (255, 165, 0))  # Orange
    meo_label = small_font.render("MEO (Orange)", True, (255, 255, 255))
    surface.blit(meo_label, (legend_x + 40, legend_y + 85))
    # GEO triangle with line break
    draw_triangle(surface, legend_x + 20, legend_y + 110, (128, 0, 128))  # Purple
    geo_label = small_font.render("GEO (Purple)", True, (255, 255, 255))
    surface.blit(geo_label, (legend_x + 40, legend_y + 105))

def draw_details(surface, hovered_satellite, selected_satellite, satellite_mean_altitudes, sub_x, sub_y, sub_width, sub_height, small_font, satellite_perigee=None, satellite_apogee=None, satellite_positions=None):
    if (hovered_satellite or selected_satellite):
        sat = selected_satellite if selected_satellite else hovered_satellite
        epoch_dt = sat.epoch.utc_datetime().strftime("%Y-%m-%d %H:%M:%S")
        details = [
            f"NORAD ID: {sat.model.satnum_str}",
            f"Name: {sat.name.strip()}",
            f"Int. Designator: {sat.model.intldesg}",
            f"Epoch: {epoch_dt}",
            f"Inclination (deg): {math.degrees(sat.model.inclo):.2f}",
            f"RAAN (deg): {math.degrees(sat.model.nodeo):.2f}",
            f"Arg. of Perigee (deg): {math.degrees(sat.model.argpo):.2f}",
            f"Mean Anomaly (deg): {math.degrees(sat.model.mo):.2f}",
            f"Mean Motion (rev/day): {sat.model.no_kozai:.4f}",
            f"Rev Number: {sat.model.revnum}"
        ]
        dist = satellite_positions[sat][3] if satellite_positions and sat in satellite_positions else None
        if ((selected_satellite or hovered_satellite) and satellite_perigee and satellite_apogee):
            details.extend([
                f"Apogee Altitude (km): {satellite_apogee[sat]:.1f}",
                f"Perigee Altitude (km): {satellite_perigee[sat]:.1f}"
            ])
        if ((selected_satellite or hovered_satellite) and dist is not None):
            details.append(f"Slant Range (km): {dist:.1f}")
        details_rect = pygame.Rect(sub_x + sub_width - 250, sub_y + 20, 230, 250)
        pygame.draw.rect(surface, (50, 50, 50), details_rect)  # Dark grey background
        pygame.draw.rect(surface, (0, 0, 0), details_rect, 2)  # Black border
        for i, line in enumerate(details):
            text_surface = small_font.render(line, True, (255, 255, 255))
            surface.blit(text_surface, (details_rect.x + 5, details_rect.y + 5 + i * 15))

def draw_time_display(surface, sub_x, sub_y, sub_height, small_font):
    current_utc = datetime.utcnow()
    current_local = current_utc - timedelta(hours=7)  # PDT is UTC-7
    utc_time_str = current_utc.strftime("%H:%M:%S.%f")[:-3]  # Millisecond precision
    local_time_str = current_local.strftime("%H:%M:%S.%f")[:-3]  # Millisecond precision
    time_text = f"UTC: {utc_time_str}  Local: {local_time_str}"
    time_surface = small_font.render(time_text, True, (255, 255, 255))
    surface.blit(time_surface, (sub_x + 10, sub_y + sub_height - 30))

def draw_satellite_count(surface, sub_x, sub_y, satellite_positions, small_font):
    sat_count = len(satellite_positions)
    count_text = f"Satellites in view: {sat_count}"
    count_surface = small_font.render(count_text, True, (255, 255, 255))
    surface.blit(count_surface, (sub_x + 10, sub_y + 200))  # Moved below filter boxes

def draw_scroll_bar(surface, scroll_bar_rect, slider_rect, small_font):
    # Draw scroll bar track
    pygame.draw.rect(surface, (100, 100, 100), scroll_bar_rect)
    # Draw slider
    pygame.draw.rect(surface, (255, 255, 255), slider_rect)
    # Draw time labels
    start_label = small_font.render("-15 min", True, (255, 255, 255))
    end_label = small_font.render("+15 min", True, (255, 255, 255))
    surface.blit(start_label, (scroll_bar_rect.x, scroll_bar_rect.y + scroll_bar_rect.height + 5))
    surface.blit(end_label, (scroll_bar_rect.x + scroll_bar_rect.width - end_label.get_width(), scroll_bar_rect.y + scroll_bar_rect.height + 5))

def draw_scroll_time_display(surface, sub_x, sub_y, sub_width, sub_height, current_tt, ts, small_font):
    radius = min(sub_width, sub_height) // 2 - 50
    current_utc = ts.tt(jd=current_tt).utc_datetime()
    current_local = current_utc - timedelta(hours=7)  # PDT is UTC-7
    utc_time_str = current_utc.strftime("%H:%M:%S.%f")[:-3]
    local_time_str = current_local.strftime("%H:%M:%S.%f")[:-3]
    time_text = f"Scroll UTC: {utc_time_str}  Local: {local_time_str}"
    time_surface = small_font.render(time_text, True, (255, 255, 255))
    text_width, _ = small_font.size(time_text)
    surface.blit(time_surface, (sub_x + sub_width // 2 - text_width // 2, sub_y + sub_height - 25))
