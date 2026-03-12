/*
 A model for a screen case for the wisecoco 10.1" capacitive touch screen
 
 wisecoco screen: https://www.amazon.com/dp/B0C7BYHN5C
 
 This code is experimental and pretty messy, full of magic numbers
 for incremental tweaks. I will likely not refactor this code to
 clean it up since I probably won't need to print another one of
 these cases.
 
 Ordinarily I wouldn't share code this gross. But, since I'm sharing
 the STL files, I thought it would be better to share this code in
 case someone really wants to mess with it. Long story short, use at
 your own risk!
*/

$fa = 2;
$fs = 0.25;

screen_length = 235;
screen_height = 142.5;
screen_thickness = 7 + 1;
screen_standoff_height = 4;
screen_standoff_offset = 5;

case_bezel_thickness = 7;
case_body_thickness = 7;
case_length = screen_length + case_bezel_thickness * 2;
case_height = screen_height + case_bezel_thickness * 2;
case_thickness = screen_thickness + screen_standoff_height + case_body_thickness;

// For M3 heatset inserts
case_mount_hole_diameter = 5.5;

// For M2.5 heatset inserts 
case_mount_screen_hole_diameter = 4.1;

case_speaker_cutout_width = 46;
case_speaker_cutout_height = 30;

back_cutout_top_offset = 44;
case_back_cutout_length = 150;
case_back_cutout_height = 85;

case_back_cover_length = case_back_cutout_length;
case_back_cover_height = case_back_cutout_height;
case_back_cover_thickness = 20 + case_body_thickness;
back_cover_mount_height = 27;
case_back_cover_length_outer = case_back_cover_length + case_body_thickness * 2;
case_back_cover_height_outer = case_back_cover_height + case_body_thickness * 2;
back_cover_mount_offset = case_back_cover_length_outer/2 + back_cover_mount_height/4;

case_back_cover_wire_cutout_length = 90;
case_back_cover_wire_cutout_depth = 15;

case_back_cover_gpio_cutout_length = 90;
case_back_cover_gpio_cutout_height = 10;

case_back_cover_usb_cutout_depth = 20;

fan_bolt_distance = 24;
fan_bolt_diameter = 3.5;
fan_center_cutout = 26;

speaker_center_offset = 60;
speaker_vertical_offset = 2;

stand_foot_length = 80;
stand_foot_height = 40;
stand_foot_thickness = 20;


////////////////////
// Rendering code //
////////////////////

// Render the main frame body
main_body();

// Render the back cover for the pi
case_back_cover();

// Render the feet
stand_feet(one_only=false);


module main_body() {
  difference() {
    // Case bezel
    //cube([case_length, case_height, case_thickness], center=true);
    roundedCube(
      case_length + 3, // 3 = bezel offset
      case_height + 2, // 2 = bezel offset
      case_thickness, 4);

    translate([0, 0, 10]) {
      back_cover_mounts(true);
    }

    // screen to subtract out
    // Add +1 to the thickness and .5 to the y-translate to make the preview
    // render nice. The cutout depth remains the same
    translate([
      0,
      0,
      case_thickness/2 - screen_thickness/2 - screen_standoff_height/2 + .5]) {
        cube([
          screen_length + 3, // 3 = bezel offset
          screen_height + 2, // 2 = bezel offset
          screen_thickness + screen_standoff_height + 1], center=true);
    }
    
    
    controls_cutout_height = screen_thickness + screen_standoff_height;
    
    // Controls cutout - Left
    
    // Audio jack cutout
    translate([
      -case_length/2 + 26/2 -2,
      -40,
      (case_thickness - controls_cutout_height)/2 + 1]) {
      cube([
        26 + 1,
        25,
        case_thickness * 2], center=true);
    }
    
    // Power and brightness cutout
    translate([
      -case_length/2,
      -12,
      (case_thickness - controls_cutout_height)/2 + 1]) {
      cube([
        75,
        75,
        controls_cutout_height + 2], center=true);
    }
    
    // Controls cutout - Right
    translate([
      case_length/2 - 25/2 +2,
      -18.5,
      (case_thickness - controls_cutout_height)/2 + 1]) {
      cube([
        25 + 1,
        66,
        case_thickness * 2], center=true);
    }
    
    // Top & bottom cutout
    top_bottom_cutout_height = screen_thickness + screen_standoff_height;
    translate([
      0,
      0,
      (case_thickness - controls_cutout_height)/2 + 1]) {
      cube([
        105,
        case_height * 2,
        controls_cutout_height + 2], center=true);
    }

    // Speaker cutouts
    translate([
      -speaker_center_offset, 
      case_height/2 - case_speaker_cutout_height/2 - case_bezel_thickness - speaker_vertical_offset,
      -case_body_thickness]) {
        cube([
          case_speaker_cutout_width,
          case_speaker_cutout_height,
          case_body_thickness * 3], center=true);
    }
    
    translate([
      speaker_center_offset, 
      case_height/2 - case_speaker_cutout_height/2 - case_bezel_thickness - speaker_vertical_offset,
      -case_body_thickness]) {
        cube([
          case_speaker_cutout_width,
          case_speaker_cutout_height,
          case_body_thickness * 3], center=true);
    }
    
    // Speaker cover mounts
    
    translate([
      speaker_center_offset + case_speaker_cutout_width/2 + 10, 
      case_height/2 - case_speaker_cutout_height/2 - case_bezel_thickness - speaker_vertical_offset,
      -case_body_thickness
    ])
      cylinder(h=case_body_thickness * 3, d=case_mount_hole_diameter, center=true);
    
    translate([
      -speaker_center_offset - case_speaker_cutout_width/2 - 10, 
      case_height/2 - case_speaker_cutout_height/2 - case_bezel_thickness - speaker_vertical_offset,
      -case_body_thickness
    ])
      cylinder(h=case_body_thickness * 3, d=case_mount_hole_diameter, center=true);
    
    // Crazy speaker cover mount because of power cord adjustments
    translate([
      -speaker_center_offset - 3, 
      case_height/2 - case_speaker_cutout_height/2 - case_bezel_thickness - speaker_vertical_offset + 20,
      -case_body_thickness
    ])
      cylinder(h=case_body_thickness, d=case_mount_hole_diameter, center=true);
      
    translate([
      speaker_center_offset + 3, 
      case_height/2 - case_speaker_cutout_height/2 - case_bezel_thickness - speaker_vertical_offset + 20,
      -case_body_thickness
    ])
      cylinder(h=case_body_thickness, d=case_mount_hole_diameter, center=true);
    
    // Power cable cutout
    translate([-31.5, 44.5, -case_thickness/2])
      cube([27, 50, case_body_thickness*3], center=true);
      
    // Power cable cutout
    translate([31.5, 44.5, -case_thickness/2])
      cube([27, 50, case_body_thickness*3], center=true);
      
    // Internal audio jack cutout
    translate([0, 35, -case_thickness/2])
      cube([40, 50, case_body_thickness*3], center=true);
    
    // Back cutout
    translate([
      0,
      (case_height - case_back_cutout_height)/2
        - case_bezel_thickness
        - back_cutout_top_offset,
      0]) {
      cube([
        case_back_cutout_length,
        case_back_cutout_height,
        case_thickness * 3], center=true);
    }
    
    // Top left hole
    translate([
      -screen_length/2 + screen_standoff_offset,
      screen_height/2 - screen_standoff_offset,
      0]) {
        cylinder(h=case_thickness * 3, d=case_mount_screen_hole_diameter, center=true);
    }
    
    // Top right hole
    translate([
      screen_length/2 - screen_standoff_offset,
      screen_height/2 - screen_standoff_offset,
      0]) {
        cylinder(h=case_thickness * 3, d=case_mount_screen_hole_diameter, center=true);
    }
    
    // Bottom right hole
    translate([
      screen_length/2 - screen_standoff_offset,
      -screen_height/2 + screen_standoff_offset,
      0]) {
        cylinder(h=case_thickness * 3, d=case_mount_screen_hole_diameter, center=true);
    }
    
    // Bottom left hole
    translate([
      -screen_length/2 + screen_standoff_offset,
      -screen_height/2 + screen_standoff_offset,
      0]) {
        cylinder(h=case_thickness * 3, d=case_mount_screen_hole_diameter, center=true);
    }
  }
}

// Case back cover

module case_back_cover() {
  translate([
    0,
    (case_height - case_back_cutout_height)/2
        - case_bezel_thickness
        - back_cutout_top_offset,
    -case_thickness/2 - case_back_cover_thickness/2]) {
    difference() {
      color("red")
        cube([
          case_back_cover_length_outer,
          case_back_cover_height_outer,
          case_back_cover_thickness], center=true);
      
      // center body cutout
      translate([0, 0, case_body_thickness/2 + .5]) {
        cube([
          case_back_cutout_length,
          case_back_cutout_height,
          case_back_cover_thickness - case_body_thickness + 1], center=true);
      }
      
      // wire cutout
      translate([
        0, 
        0,
        case_back_cover_thickness/2 - case_back_cover_wire_cutout_depth/2])
        cube([
          case_back_cover_wire_cutout_length,
          case_back_cover_height_outer * 2,
          case_back_cover_wire_cutout_depth], center=true);
      
      // Fan cutout
      translate([
        0,
        0,
        -case_back_cover_thickness/2])
        cylinder(h=case_body_thickness*3, d=fan_center_cutout, center=true);
      
      // Fan cutout mount holes    
      translate([
        fan_bolt_distance/2,
        fan_bolt_distance/2,
        -case_back_cover_thickness/2])
          cylinder(h=case_body_thickness*3, d=fan_bolt_diameter, center=true);
      
      translate([
        -fan_bolt_distance/2,
        fan_bolt_distance/2,
        -case_back_cover_thickness/2])
          cylinder(h=case_body_thickness*3, d=fan_bolt_diameter, center=true);
          
      translate([
        -fan_bolt_distance/2,
        -fan_bolt_distance/2,
        -case_back_cover_thickness/2])
          cylinder(h=case_body_thickness*3, d=fan_bolt_diameter, center=true);
          
      translate([
        fan_bolt_distance/2,
        -fan_bolt_distance/2,
        -case_back_cover_thickness/2])
          cylinder(h=case_body_thickness*3, d=fan_bolt_diameter, center=true);
      
      // USB cutout
      translate([
        case_back_cover_length_outer/2, 
        0,
        4])
        cube([
          case_back_cover_wire_cutout_length,
          case_back_cover_height_outer / 3 + 22,
          case_back_cover_usb_cutout_depth], center=true);

      // GIPO cutout   
      translate([
        0,
        -case_back_cover_height/2 + case_body_thickness + 4,
        -case_back_cover_thickness/2])
          cube([
            case_back_cover_gpio_cutout_length,
            case_back_cover_gpio_cutout_height,
            case_body_thickness*3], center=true);
      
      // Foot mount holes
      translate([
        case_back_cover_length/2 - case_body_thickness/2 - 10,
        -case_back_cover_height/2 - case_body_thickness + 15,
        -case_back_cover_thickness/2])
          cylinder(h=case_body_thickness*20, d=case_mount_hole_diameter, center=true);
          
      translate([
        -case_back_cover_length/2 + case_body_thickness/2 + 10,
        -case_back_cover_height/2 - case_body_thickness + 15,
        -case_back_cover_thickness/2])
          cylinder(h=case_body_thickness*20, d=case_mount_hole_diameter, center=true);
          
      translate([
        case_back_cover_length/2 - case_body_thickness/2 - 10,
        -case_back_cover_height/2 - case_body_thickness + 10,
        -case_back_cover_thickness/2 + 15])
          rotate([90, 0, 0])
            cylinder(h=case_body_thickness*20, d=case_mount_hole_diameter, center=true);

      translate([
        -case_back_cover_length/2 + case_body_thickness/2 + 10,
        -case_back_cover_height/2 - case_body_thickness + 10,
        -case_back_cover_thickness/2 + 15])
          rotate([90, 0, 0])
            cylinder(h=case_body_thickness*20, d=case_mount_hole_diameter, center=true);
    }
  }

  // Case back cover mounts
  back_cover_mounts();
}

module back_cover_mounts(holes_only=false) {

  translate([
    -back_cover_mount_offset,
    23.5,
    -back_cover_mount_height/2 - case_body_thickness - 2.5])
      if (!holes_only) {
        back_cover_mount(back_cover_mount_height, case_mount_hole_diameter);
      } else {
        cylinder(h=case_body_thickness * 20, d=case_mount_hole_diameter, center=true);
      }

  translate([
    -back_cover_mount_offset,
    -55,
    -back_cover_mount_height/2 - case_body_thickness - 2.5])
      if (!holes_only) {
        back_cover_mount(back_cover_mount_height, case_mount_hole_diameter);
      } else {
        cylinder(h=case_body_thickness * 20, d=case_mount_hole_diameter, center=true);
      }

  translate([
    back_cover_mount_offset,
    23.5,
    -back_cover_mount_height/2 - case_body_thickness - 2.5])
      if (!holes_only) {
        rotate([0, 0, 180])
            back_cover_mount(back_cover_mount_height, case_mount_hole_diameter);
      } else {
        cylinder(h=case_body_thickness * 20, d=case_mount_hole_diameter, center=true);
      }

  translate([
    back_cover_mount_offset,
    -55,
    -back_cover_mount_height/2 - case_body_thickness - 2.5])
      if (!holes_only) {
        rotate([0, 0, 180])
          back_cover_mount(back_cover_mount_height, case_mount_hole_diameter);
      } else {
        cylinder(h=case_body_thickness * 20, d=case_mount_hole_diameter, center=true);
      }
}

module back_cover_mount(height, hole_diameter) {
  cube_size = height / 2;
  
  translate([0, 0, cube_size / 2]) {
    difference() {
      cube([cube_size, cube_size, cube_size], center=true);
      cylinder(h=cube_size/2 + 1, d=hole_diameter);
    }
    
    translate([0, 0, -cube_size]) {
      rotate([0, -45, 0]) {
        difference() {
          rotate([0, 45, 0]) cube([cube_size, cube_size, cube_size], center=true);
          
          translate([-cube_size, 0, 0]) {
            cube([cube_size * 2, cube_size * 2, cube_size * 2], center=true);
          }
        }
      }
    }
  }
}

// Stand feet
module stand_feet(one_only=true) {
  translate([
    case_back_cover_length/2 - case_body_thickness/2 - 10,
    -case_back_cover_height/2 - case_body_thickness/2 - 5,
    -case_thickness/2 -case_back_cover_thickness - stand_foot_length/2 + 20])
      rotate([0, 90, 0])
        stand_foot();

  if (!one_only) { 
      translate([
        -case_back_cover_length/2 + case_body_thickness/2 + 10,
        -case_back_cover_height/2 - case_body_thickness/2 - 5,
        -case_thickness/2 -case_back_cover_thickness - stand_foot_length/2 + 20])
          rotate([0, 90, 0])
            stand_foot();
  }
}

module stand_foot() {
  chopped_tip_size = 4;
  cutout_offset_y = 8;
  cutout_offset_x = 20;
 
  points = [
      [-5, 0],
      [-5, cutout_offset_y],
      [cutout_offset_x, cutout_offset_y],
      [cutout_offset_x, stand_foot_height],
      [stand_foot_length, 4],
      [stand_foot_length, 0],
  ];

  difference() {
    translate([-stand_foot_length/2, -stand_foot_height/2, -stand_foot_thickness/2])
      linear_extrude(height = stand_foot_thickness) {
        polygon(points);
      }
    
    // bottom hole
    translate([-stand_foot_length/2 + 5, 0, 0])
      rotate([90, 0, 0])
        cylinder(h=stand_foot_length, d=case_mount_hole_diameter);
      
      
    // horizontal hole
    translate([
        -stand_foot_length/2,
        -stand_foot_height/2 + cutout_offset_y + 15,
        0])
      rotate([0, 90, 0])
        cylinder(h=stand_foot_length, d=case_mount_hole_diameter);
  }
}

module inverseCorner(radius, length) {
    difference() {
        cube([radius + 0.01, radius + 0.01, length], center = true);

        translate([radius / 2, radius / 2, 0])
            cylinder(r = radius, h = length + 1, center = true);
    }
}

module roundedCube(width, depth, height, radius) {
    // Create a rounded cube using hull of spheres at the corners
    hull() {
        // Bottom corners
        translate([-width/2 + radius, -depth/2 + radius, -height/2 + radius])
            sphere(r = radius);
        translate([width/2 - radius, -depth/2 + radius, -height/2 + radius])
            sphere(r = radius);
        translate([-width/2 + radius, depth/2 - radius, -height/2 + radius])
            sphere(r = radius);
        translate([width/2 - radius, depth/2 - radius, -height/2 + radius])
            sphere(r = radius);
        
        // Top corners
        translate([-width/2 + radius, -depth/2 + radius, height/2 - radius])
            sphere(r = radius);
        translate([width/2 - radius, -depth/2 + radius, height/2 - radius])
            sphere(r = radius);
        translate([-width/2 + radius, depth/2 - radius, height/2 - radius])
            sphere(r = radius);
        translate([width/2 - radius, depth/2 - radius, height/2 - radius])
            sphere(r = radius);
    }
}